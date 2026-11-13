import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone

class Store:
    def __init__(self, path):
        self.path=str(path)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS comments(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, comment_id TEXT NOT NULL, recommendation TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tool_events(run_id TEXT NOT NULL, seq INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(run_id,seq));
            CREATE TABLE IF NOT EXISTS overrides(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), decision TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path)
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_comment(self, comment):
        cid=comment.identity()
        payload=comment.model_dump(); payload['id']=cid
        encoded=json.dumps(payload,ensure_ascii=False,sort_keys=True)
        with self.connect() as db:
            previous=db.execute('SELECT payload FROM comments WHERE id=?',(cid,)).fetchone()
            if previous and previous[0]!=encoded:
                raise ValueError('Comment ID already belongs to different text or context')
            db.execute('INSERT OR IGNORE INTO comments VALUES (?,?)',(cid,encoded))
        return cid

    def context(self, cid):
        with self.connect() as db:
            row=db.execute('SELECT payload FROM comments WHERE id=?',(cid,)).fetchone()
        value=json.loads(row[0]) if row else {}
        # Never look up an unrelated globally supplied parent_id.
        return {'available':bool(value.get('parent_text')), 'parent_id':value.get('parent_id'), 'text':value.get('parent_text')}

    def save(self, result):
        with self.connect() as db:
            db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',(result['run_id'],result['comment_id'],result['recommendation'],json.dumps(result,ensure_ascii=False),datetime.now(timezone.utc).isoformat()))
            db.executemany('INSERT INTO tool_events VALUES (?,?,?)',[(result['run_id'],i,json.dumps(e,ensure_ascii=False)) for i,e in enumerate(result['tool_events'])])

    def override(self, run_id, decision, note=''):
        if decision not in ('FLAG','NO_ISSUE_DETECTED','HUMAN_REVIEW'):
            raise ValueError('Invalid decision')
        with self.connect() as db:
            db.execute('INSERT INTO overrides(run_id,decision,note,created_at) VALUES (?,?,?,?)',(run_id,decision,note,datetime.now(timezone.utc).isoformat()))

    def results(self, limit=None, recommendation=None):
        """Newest runs first. ``limit`` and ``recommendation`` keep the UI from loading the whole history."""
        query='SELECT run_id,payload FROM runs'; params=[]
        if recommendation:
            query+=' WHERE recommendation=?'; params.append(recommendation)
        query+=' ORDER BY created_at DESC'
        if limit is not None:
            query+=' LIMIT ?'; params.append(int(limit))
        with self.connect() as db:
            rows=db.execute(query,params).fetchall()
            overrides=db.execute('SELECT run_id,decision,note,created_at FROM overrides ORDER BY id').fetchall()
        latest={r[0]:dict(decision=r[1],note=r[2],created_at=r[3]) for r in overrides}
        return [dict(json.loads(payload),human_override=latest.get(rid)) for rid,payload in rows]

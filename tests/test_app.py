from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_application_screens_render():
    app=AppTest.from_file(str(Path(__file__).parents[1]/'app.py'),default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value('Review queue').run()
    assert not app.exception
    app.sidebar.radio[0].set_value('Evaluation').run()
    assert not app.exception

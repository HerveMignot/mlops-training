import pandas as pd

from training.data import load_data


def test_load_data(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")

    df = load_data(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2

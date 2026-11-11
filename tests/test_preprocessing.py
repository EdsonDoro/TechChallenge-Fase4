import pandas as pd
from src.data.preprocessing import prepare_reviews

def test_prepare_reviews_removes_empty_comments():
    df = pd.DataFrame({
        "review_id": ["1", "2"],
        "review_comment_message": ["Entrega rápida", None],
    })
    result = prepare_reviews(df)
    assert len(result) == 1
    assert result.iloc[0]["review_comment_message"] == "Entrega rápida"

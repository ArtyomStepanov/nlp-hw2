import datasets as ds
import pytest

from prepare_datasets import remove_code_comments, remove_dataset_comments


@pytest.mark.parametrize("name", ["paint＿container", "cafe\u0301", "paint‿container"])
def test_unicode_identifier_preserved_when_removing_comments(name):
    code = (
        "def solve():\n"
        f"    {name} = 4  # количество\n"
        '    text = "# строка"\n'
        '    doc = """# тоже строка"""\n'
        f"    return {name}  # результат\n"
    )
    expected = (
        "def solve():\n"
        f"    {name} = 4  \n"
        '    text = "# строка"\n'
        '    doc = """# тоже строка"""\n'
        f"    return {name}  \n"
    )
    assert remove_code_comments(code) == expected


def test_dataset_comments_support_unicode_with_multiple_processes():
    original = ds.DatasetDict({
        "train": ds.Dataset.from_dict({
            "question": ["first", "second"],
            "code": ["paint＿container = 4 # comment\n", None],
        }),
        "test": ds.Dataset.from_dict({"question": ["official test"]}),
    })
    cleaned = remove_dataset_comments(original, num_proc=2, batch_size=1)
    assert cleaned["train"]["code"] == ["paint＿container = 4 \n", None]
    assert cleaned["train"]["question"] == ["first", "second"]
    assert cleaned["test"].to_dict() == {"question": ["official test"]}
    assert original["train"]["code"] == ["paint＿container = 4 # comment\n", None]


@pytest.mark.parametrize("code", [
    'x = "unterminated # this is not a comment\n',
    'paint＿container = "unterminated # this is not a comment\n',
    "x = (1 # unterminated expression\n",
    'x = """unterminated\n',
])
def test_malformed_code_still_raises(code):
    with pytest.raises(ValueError, match="Не удалось токенизировать код"):
        remove_code_comments(code)

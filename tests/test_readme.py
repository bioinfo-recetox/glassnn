"""The 5-minute tour in README.md runs, and prints what it says it prints."""

import doctest
import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_the_readme_tour_is_a_passing_doctest():
    # The ```pycon blocks, run in order in one namespace, as one doctest.
    blocks = re.findall(r"```pycon\n(.*?)```", README.read_text(), flags=re.DOTALL)
    test = doctest.DocTestParser().get_doctest(
        "\n".join(blocks), {}, "README.md", str(README), 0
    )
    runner = doctest.DocTestRunner(
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )
    runner.run(test)
    result = runner.summarize(verbose=False)
    assert result.attempted >= 15, "the tour should have at least 15 examples"
    assert result.failed == 0

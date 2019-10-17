import parser


def test_parse_file():
    directives = parser.parse_file("test_fixtures/sample_program.untitledlanguage")

    assert directives["from_data"] == "test_fixtures/credit_data.csv"

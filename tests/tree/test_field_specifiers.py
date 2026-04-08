import attrs
import jax

import jarp


@jarp.define
class Example:
    data: object = jarp.array(default=[1, 2])
    meta: str = jarp.static(default="x")
    maybe: object = jarp.auto(default="y")


def test_field_specifiers_store_expected_metadata() -> None:
    fields = attrs.fields_dict(Example)
    assert "static" not in fields["data"].metadata
    assert fields["meta"].metadata["static"] is True
    assert fields["maybe"].metadata["static"] is jarp.tree.FieldType.AUTO
    assert isinstance(Example().data, jax.Array)


def test_enum_coercions_match_jax_conventions() -> None:
    assert bool(jarp.tree.FieldType.META)
    assert not bool(jarp.tree.FieldType.DATA)
    assert jarp.tree.FieldType(True) is jarp.tree.FieldType.META
    assert jarp.tree.FieldType(None) is jarp.tree.FieldType.DATA
    assert jarp.tree.PyTreeType(False) is jarp.tree.PyTreeType.NONE
    assert jarp.tree.PyTreeType("static") is jarp.tree.PyTreeType.STATIC

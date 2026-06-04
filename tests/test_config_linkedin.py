from app.config import Settings


def test_linkedin_credential_fields_declared_optional_with_guest_defaults():
    # Assert the DECLARED defaults, not the loaded singleton (which reflects any
    # real .env values). Absent config must mean "no Voyager, guest fallback".
    fields = Settings.model_fields
    assert fields["linkedin_email"].default == ""
    assert fields["linkedin_password"].default == ""
    assert fields["linkedin_storage_state_path"].default == "./data/linkedin_state.json"

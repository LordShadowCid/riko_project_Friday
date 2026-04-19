from server.process.memory.bio_manager import ensure_speaker, add_fact, update_bio, get_bio
ensure_speaker('Dad')
update_bio('Dad', 'relationship', 'primary_user')
add_fact('Dad', 'likes coffee')
add_fact('Dad', 'has a daughter named Riley')
result = get_bio('Dad')
print(result)
assert 'primary_user' in result, 'relationship missing'
assert 'coffee' in result, 'fact missing'
print('PASS')

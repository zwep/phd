
# Function to strip file name from a path
strip_filename() {
  path=$1
  dirname="$(dirname "$path")"
  echo "$dirname"
}

strip_dirname() {
  path=$1
  dirname="$(basename "$path")"
  echo "$dirname"
}

# Help message
display_help() {
  echo "Usage: script_name.sh [options]"
  echo "    Not doing anything"
}

# Used to get the proper model number..
find_model_file() {
  local model_dir="$1"
  local latest_file=""

  for file in "${model_dir}"/model_*.pt; do
    if [[ -z "$latest_file" || "$file" -nt "$latest_file" ]]; then
      latest_file="$file"
    fi
  done

  # This is a new way of selecting the appropriate model
  # Adding deprecated to this call makes sure that we never select it..
  local model_file_nr="${model_dir}/model_number_deprecated"

  if [[ -f $model_file_nr ]]; then
    read -r found_number < "${model_file_nr}"
    #echo -e "Select model .pt file number ${found_number}"
    local checkpoint_path="${model_dir}/model_${found_number}.pt"
  else
    local checkpoint_path="${latest_file}"
  fi
  # Only numeric values can be returned
  # Use echo for strings, but dont echo anything else in the function
  echo "$checkpoint_path"
}

#!/bin/bash

show_spinner() {
  local -r pid="$1"
  local -r delay='0.1'
  local spinstr='|/-\\'

  while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
    local temp=${spinstr#?}
    printf " [%c]  " "$spinstr"
    local spinstr=$temp${spinstr%"$temp"}
    sleep $delay
    printf "\b\b\b\b\b\b"
  done
  printf "    \b\b\b\b"
}

# Evaluate the command provided as the first argument
# Surrounding $1 with quotes ensures it's handled as a single string
eval "$1" &

# Get the PID of the command
pid=$!

# Show the spinner until the command finishes.
show_spinner "$pid"

wait $pid
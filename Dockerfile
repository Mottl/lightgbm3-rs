FROM rust
RUN apt update
RUN apt install -y cmake libclang-dev libc++-dev gcc-multilib
WORKDIR /app

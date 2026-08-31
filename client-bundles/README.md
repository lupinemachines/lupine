# Server client bundles

CI assembles the native client objects into this directory before the server
publish workflow builds an image. That workflow sets
`LUPINE_REQUIRE_CLIENT_BUNDLES=1`, so the image build fails when any variant is
absent. Development and compile-check server images may omit the bundles; the
server then answers the client endpoint with `503`.

# Server client bundles

CI assembles the native client objects into this directory before building a
production server image; the server image target intentionally fails when any
variant is absent. A development server binary with no configured bundle tree
answers the client endpoint with `503`.

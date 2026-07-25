{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python313
    uv
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib   # libstdc++ (numpy, scipy, torch, etc.)
    postgresql.lib      # libpq (psycopg2)
    openssl             # SSL (asyncpg, httpx, etc.)
    zlib                # compression (pandas, pillow, etc.)
  ]);

  shellHook = ''
    export UV_PYTHON_PREFERENCE=only-system
  '';
}

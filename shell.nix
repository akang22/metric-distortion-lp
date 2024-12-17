let
  pkgs = import <nixpkgs> {};
  python = pkgs.python3.override {
    self = python;
    packageOverrides = pyfinal: pyprev: {
    };
  };
in pkgs.mkShell {
  packages = [
    (python.withPackages (python-pkgs: [
      python-pkgs.pulp
      python-pkgs.tqdm
      python-pkgs.matplotlib
    ]))
  ];
}


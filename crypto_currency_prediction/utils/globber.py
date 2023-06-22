from glob import glob


def globber(pattern: str) -> str:
  '''Find the path of the most recent file in a local directory using a given pattern.'''
  paths = glob(pathname=pattern)
  if paths == []:
    raise RuntimeError
  return paths[-1]

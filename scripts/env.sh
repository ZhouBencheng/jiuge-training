#!/usr/bin/env bash

_corex_lib=/usr/local/corex/lib64
case ":${LD_LIBRARY_PATH-}:" in
  *":${_corex_lib}:"*) ;;
  *) export LD_LIBRARY_PATH="${_corex_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
unset _corex_lib

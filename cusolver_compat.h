#ifndef LUPINE_CUSOLVER_COMPAT_H
#define LUPINE_CUSOLVER_COMPAT_H

// The shim forwards the deprecated calls too.
#define DISABLE_CUSOLVER_DEPRECATED
#define DISABLE_CUSOLVERMG_DEPRECATED
#include <cusolverDn.h>
#include <cusolverMg.h>
#include <cusolverRf.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#endif

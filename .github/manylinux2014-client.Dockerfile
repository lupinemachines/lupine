ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux2014_x86_64:latest
FROM ${MANYLINUX_IMAGE}

ARG NGHTTP2_VERSION=1.70.0
ARG NGHTTP2_SHA256=aa317e2cf9dca6afa0aed68f8fad6ff303ec6982e25a78c75c0b65e2b9b3ded5
ARG OPENSSL_VERSION=3.5.8
ARG OPENSSL_SHA256=a8f84a39918ec6415ce765d9b429d313ba97b8143169c172e734b9514464f5b2

ENV PATH="/opt/python/cp312-cp312/bin:${PATH}"
ENV PERL5LIB="/opt/lupine-build-perl"

COPY perl/IPC/Cmd.pm /opt/lupine-build-perl/IPC/Cmd.pm

# OpenSSL's generated Makefile uses Time::Piece. Install the matching CentOS 7
# module directly from the archival repository so the build does not depend on
# the retired distribution's yum metadata remaining available.
RUN set -eux; \
    case "$(uname -m)" in \
      x86_64) \
        rpm_url=https://archive.kernel.org/centos-vault/7.9.2009/updates/x86_64/Packages/perl-Time-Piece-1.20.1-299.el7_9.x86_64.rpm; \
        rpm_sha256=16c4eac47cfb42c7b2453290f21172024727ffd7444425d48a2170493b43db78 \
        ;; \
      aarch64) \
        rpm_url=https://archive.kernel.org/centos-vault/altarch/7.9.2009/updates/aarch64/Packages/perl-Time-Piece-1.20.1-299.el7_9.aarch64.rpm; \
        rpm_sha256=b50ce810ce320041475319673162e1d9aeabd9cbf86407567ad29f5d0d5d1df2 \
        ;; \
      *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    curl -fL "$rpm_url" -o /tmp/perl-Time-Piece.rpm; \
    echo "$rpm_sha256  /tmp/perl-Time-Piece.rpm" | sha256sum -c -; \
    rpm -Uvh /tmp/perl-Time-Piece.rpm; \
    rm -f /tmp/perl-Time-Piece.rpm

# The client shims are loaded into arbitrary host processes, so all private
# dependencies are PIC static archives built against the manylinux2014 ABI.
# glibc itself remains dynamic and is capped at the image's GLIBC_2.17 floor.
RUN set -eux; \
    curl -fL \
      "https://github.com/nghttp2/nghttp2/releases/download/v${NGHTTP2_VERSION}/nghttp2-${NGHTTP2_VERSION}.tar.gz" \
      -o /tmp/nghttp2.tar.gz; \
    echo "${NGHTTP2_SHA256}  /tmp/nghttp2.tar.gz" | sha256sum -c -; \
    tar -C /tmp -xzf /tmp/nghttp2.tar.gz; \
    cd "/tmp/nghttp2-${NGHTTP2_VERSION}"; \
    CFLAGS=-fPIC ./configure \
      --prefix=/opt/lupine-deps \
      --libdir=/opt/lupine-deps/lib \
      --enable-lib-only \
      --enable-static \
      --disable-shared; \
    make -s -j"$(getconf _NPROCESSORS_ONLN)"; \
    make -s install; \
    rm -rf /tmp/nghttp2.tar.gz "/tmp/nghttp2-${NGHTTP2_VERSION}"

RUN set -eux; \
    curl -fL \
      "https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz" \
      -o /tmp/openssl.tar.gz; \
    echo "${OPENSSL_SHA256}  /tmp/openssl.tar.gz" | sha256sum -c -; \
    tar -C /tmp -xzf /tmp/openssl.tar.gz; \
    cd "/tmp/openssl-${OPENSSL_VERSION}"; \
    case "$(uname -m)" in \
      x86_64) openssl_target=linux-x86_64 ;; \
      aarch64) openssl_target=linux-aarch64 ;; \
      *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    perl ./Configure "$openssl_target" \
      --prefix=/opt/lupine-deps \
      --openssldir=/etc/ssl \
      --libdir=lib \
      no-module \
      no-shared \
      no-tests \
      no-zlib \
      -fPIC; \
    make -s -j"$(getconf _NPROCESSORS_ONLN)"; \
    make -s install_sw; \
    rm -rf /tmp/openssl.tar.gz "/tmp/openssl-${OPENSSL_VERSION}"

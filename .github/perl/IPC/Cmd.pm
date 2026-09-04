package IPC::Cmd;

use strict;
use warnings;
use File::Spec;

# OpenSSL's platform detection only needs can_run(). CentOS 7 split IPC::Cmd
# and its large dependency chain out of the Perl base package, so keep this
# build-only implementation local to the manylinux image.
sub can_run {
    my ($command) = @_;
    return unless defined $command && length $command;
    return $command if File::Spec->file_name_is_absolute($command) && -x $command;
    for my $directory (File::Spec->path()) {
        my $candidate = File::Spec->catfile($directory, $command);
        return $candidate if -f $candidate && -x $candidate;
    }
    return;
}

1;

/* Compiled Linux wait/wake semantics using native pthread primitives.
 * This backend supports waits within the current process, not shared mappings
 * across independently running processes or priority-inheritance operations. */
struct aot_waiter {
    struct aot_waiter *next;
    uint32_t *address;
    uint32_t bitset;
    int awakened;
    pthread_cond_t condition;
};
static pthread_mutex_t aot_wait_lock=PTHREAD_MUTEX_INITIALIZER;
static struct aot_waiter *aot_waiters;
static void aot_remove_waiter(struct aot_waiter *waiter) {
    struct aot_waiter **entry=&aot_waiters;
    while (*entry && *entry!=waiter) entry=&(*entry)->next;
    if (*entry) *entry=waiter->next;
}
static void aot_cancel_waiter(void *argument) {
    struct aot_waiter *waiter=argument;
    aot_remove_waiter(waiter);
    pthread_mutex_unlock(&aot_wait_lock);
    pthread_cond_destroy(&waiter->condition);
}
static long aot_futex(long pointer,long operation,long value,long timeout,long other,long bitset) {
    (void)other;
    if (!pointer) return -14;
    if (pointer & 3) return -22;
    if (operation & ~0x1ffL) return -38;
    int command=(int)operation & 0x7f;
    if (command!=0 && command!=1 && command!=9 && command!=10) return -38;
    uint32_t mask=(command==9 || command==10) ? (uint32_t)bitset : UINT32_MAX;
    if (!mask) return -22;
    uint32_t *address=(void *)pointer;
    pthread_mutex_lock(&aot_wait_lock);
    if (command==1 || command==10) {
        if ((int)value<0) {pthread_mutex_unlock(&aot_wait_lock);return -22;}
        long count=0;
        for (struct aot_waiter *waiter=aot_waiters;waiter && count<(int)value;waiter=waiter->next) {
            if (waiter->address==address && !waiter->awakened && (waiter->bitset & mask)) {
                waiter->awakened=1;
                pthread_cond_signal(&waiter->condition);
                ++count;
            }
        }
        pthread_mutex_unlock(&aot_wait_lock);
        return count;
    }
    if (__atomic_load_n(address,__ATOMIC_SEQ_CST)!=(uint32_t)value) {
        pthread_mutex_unlock(&aot_wait_lock);return -11;
    }
    struct timespec deadline={0,0};
    clockid_t clock=(operation & 0x100) ? CLOCK_REALTIME : CLOCK_MONOTONIC;
    if (timeout) {
        const struct linux_timespec *input=(void *)timeout;
        if (input->seconds<0 || input->nanoseconds<0 || input->nanoseconds>=1000000000) {
            pthread_mutex_unlock(&aot_wait_lock);return -22;
        }
        deadline.tv_sec=input->seconds;
        deadline.tv_nsec=input->nanoseconds;
        if (command==0) {
            struct timespec now;
            clock_gettime(clock,&now);
            deadline.tv_sec+=now.tv_sec;
            deadline.tv_nsec+=now.tv_nsec;
            if (deadline.tv_nsec>=1000000000) {++deadline.tv_sec;deadline.tv_nsec-=1000000000;}
        }
    }
    struct aot_waiter waiter={.next=aot_waiters,.address=address,.bitset=mask,.awakened=0};
    int error=pthread_cond_init(&waiter.condition,NULL);
    if (error) {pthread_mutex_unlock(&aot_wait_lock);return -aot_linux_errno(error);}
    aot_waiters=&waiter;
    pthread_cleanup_push(aot_cancel_waiter,&waiter);
    while (!waiter.awakened) {
        if (!timeout) error=pthread_cond_wait(&waiter.condition,&aot_wait_lock);
        else {
            struct timespec now,remaining;
            clock_gettime(clock,&now);
            remaining.tv_sec=deadline.tv_sec-now.tv_sec;
            remaining.tv_nsec=deadline.tv_nsec-now.tv_nsec;
            if (remaining.tv_nsec<0) {--remaining.tv_sec;remaining.tv_nsec+=1000000000;}
            if (remaining.tv_sec<0) {error=ETIMEDOUT;break;}
            error=pthread_cond_timedwait_relative_np(&waiter.condition,&aot_wait_lock,&remaining);
        }
        if (error) break;
    }
    pthread_cleanup_pop(0);
    aot_remove_waiter(&waiter);
    int awakened=waiter.awakened;
    pthread_mutex_unlock(&aot_wait_lock);
    pthread_cond_destroy(&waiter.condition);
    return awakened ? 0 : error ? -aot_linux_errno(error) : 0;
}

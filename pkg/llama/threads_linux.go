package llama

import (
	"fmt"
	"io/fs"
	"os"
	"strconv"
	"strings"
)

func mathCores() int { return countMathCores(os.DirFS("/sys/devices")) }

// countMathCores counts physical performance cores from a sysfs tree rooted at
// /sys/devices. Hybrid Intel parts list their P-cores in cpu_core/cpus; on
// anything else every online CPU counts. Each core is counted once by its
// thread_siblings_list, so SMT siblings collapse into one entry. It returns 0
// when sysfs is unreadable, which clampThreads treats as unknown.
func countMathCores(sys fs.FS) int {
	cpus, err := readCPUList(sys, "cpu_core/cpus")
	if err != nil {
		cpus, err = readCPUList(sys, "system/cpu/online")
	}
	if err != nil {
		return 0
	}
	cores := make(map[string]struct{}, len(cpus))
	for _, cpu := range cpus {
		b, err := fs.ReadFile(sys, fmt.Sprintf("system/cpu/cpu%d/topology/thread_siblings_list", cpu))
		if err != nil {
			continue
		}
		cores[strings.TrimSpace(string(b))] = struct{}{}
	}
	return len(cores)
}

func readCPUList(sys fs.FS, name string) ([]int, error) {
	b, err := fs.ReadFile(sys, name)
	if err != nil {
		return nil, err
	}
	return parseCPUList(string(b))
}

// parseCPUList parses the kernel's cpulist format, for example "0-3,8,10-11".
func parseCPUList(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, fmt.Errorf("empty cpu list")
	}
	var cpus []int
	for part := range strings.SplitSeq(s, ",") {
		lo, hi, isRange := strings.Cut(part, "-")
		first, err := strconv.Atoi(lo)
		if err != nil {
			return nil, fmt.Errorf("cpu list %q: %w", s, err)
		}
		last := first
		if isRange {
			if last, err = strconv.Atoi(hi); err != nil {
				return nil, fmt.Errorf("cpu list %q: %w", s, err)
			}
		}
		if last < first {
			return nil, fmt.Errorf("cpu list %q: range %s runs backwards", s, part)
		}
		for cpu := first; cpu <= last; cpu++ {
			cpus = append(cpus, cpu)
		}
	}
	return cpus, nil
}

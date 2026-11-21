package llama

import (
	"fmt"
	"slices"
	"testing"
	"testing/fstest"
)

func TestParseCPUList(t *testing.T) {
	tests := []struct {
		in      string
		want    []int
		wantErr bool
	}{
		{"0-3\n", []int{0, 1, 2, 3}, false},
		{"0,2,4-5", []int{0, 2, 4, 5}, false},
		{"7", []int{7}, false},
		{"", nil, true},
		{"3-1", nil, true},
		{"a-b", nil, true},
	}
	for _, tt := range tests {
		got, err := parseCPUList(tt.in)
		if (err != nil) != tt.wantErr {
			t.Errorf("parseCPUList(%q) error = %v, wantErr %v", tt.in, err, tt.wantErr)
			continue
		}
		if !slices.Equal(got, tt.want) {
			t.Errorf("parseCPUList(%q) = %v, want %v", tt.in, got, tt.want)
		}
	}
}

// sysfs builds a fake /sys/devices tree. siblings maps each CPU to its
// thread_siblings_list.
func sysfs(online, pcores string, siblings map[int]string) fstest.MapFS {
	fsys := fstest.MapFS{"system/cpu/online": {Data: []byte(online + "\n")}}
	if pcores != "" {
		fsys["cpu_core/cpus"] = &fstest.MapFile{Data: []byte(pcores + "\n")}
	}
	for cpu, sib := range siblings {
		fsys[sysfsSiblings(cpu)] = &fstest.MapFile{Data: []byte(sib + "\n")}
	}
	return fsys
}

func sysfsSiblings(cpu int) string {
	return fmt.Sprintf("system/cpu/cpu%d/topology/thread_siblings_list", cpu)
}

func TestCountMathCores(t *testing.T) {
	// The i5-12500H layout: CPUs 0-7 are four SMT P-cores, 8-15 are E-cores.
	alderLake := map[int]string{
		0: "0-1", 1: "0-1", 2: "2-3", 3: "2-3", 4: "4-5", 5: "4-5", 6: "6-7", 7: "6-7",
		8: "8", 9: "9", 10: "10", 11: "11", 12: "12", 13: "13", 14: "14", 15: "15",
	}
	tests := []struct {
		name string
		fsys fstest.MapFS
		want int
	}{
		{"hybrid counts P-cores once each", sysfs("0-15", "0-7", alderLake), 4},
		{"non-hybrid SMT counts physical cores", sysfs("0-7", "", map[int]string{
			0: "0,4", 1: "1,5", 2: "2,6", 3: "3,7", 4: "0,4", 5: "1,5", 6: "2,6", 7: "3,7",
		}), 4},
		{"no SMT counts every cpu", sysfs("0-3", "", map[int]string{0: "0", 1: "1", 2: "2", 3: "3"}), 4},
		{"no sysfs is unknown", fstest.MapFS{}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := countMathCores(tt.fsys); got != tt.want {
				t.Errorf("countMathCores = %d, want %d", got, tt.want)
			}
		})
	}
}

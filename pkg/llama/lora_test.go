package llama

import (
	"errors"
	"testing"
)

// Loading a real adapter needs a LoRA GGUF and its matching base model, which the
// test suite does not ship, so these tests cover the guards instead: a missing
// file, a closed model, and the bookkeeping that stops an adapter being dropped
// while a context still uses it.

func TestLoadLoraRejectsMissingFile(t *testing.T) {
	m := &Model{path: "fake", ptr: nil}
	_, err := m.LoadLora("/nonexistent/adapter.gguf")
	if !errors.Is(err, ErrModelLoad) {
		t.Fatalf("LoadLora = %v, want ErrModelLoad", err)
	}
}

func TestLoadLoraRejectsClosedModel(t *testing.T) {
	m := &Model{path: "fake"}
	m.closed.Store(true)
	_, err := m.LoadLora("/tmp/whatever.gguf")
	if !errors.Is(err, ErrClosed) {
		t.Fatalf("LoadLora on a closed model = %v, want ErrClosed", err)
	}
}

func TestLoraAdapterNilSafety(t *testing.T) {
	var a *LoraAdapter
	if err := a.Close(); err != nil {
		t.Fatalf("nil adapter Close = %v", err)
	}
	if got := a.Applied(); got != 0 {
		t.Fatalf("nil adapter Applied = %d", got)
	}
	if _, ok := a.MetaValue("general.name"); ok {
		t.Fatal("nil adapter returned metadata")
	}
	if err := a.Apply(nil, 1); !errors.Is(err, ErrClosed) {
		t.Fatalf("Apply on nil adapter = %v, want ErrClosed", err)
	}
}

func TestLoraCloseRefusesWhileApplied(t *testing.T) {
	a := &LoraAdapter{path: "adapter.gguf", applied: 1}
	if err := a.Close(); err == nil {
		t.Fatal("Close succeeded while the adapter was applied")
	}
	a.applied = 0
	if err := a.Close(); err != nil {
		t.Fatalf("Close after removing: %v", err)
	}
	if err := a.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestLoraApplyChecksModelIdentity(t *testing.T) {
	owner := &Model{path: "owner"}
	other := &Model{path: "other"}
	a := &LoraAdapter{path: "adapter.gguf", model: owner}
	c := &Context{model: other}
	if err := a.Apply(c, 1); err == nil {
		t.Fatal("Apply accepted a context from a different model")
	}
}

func TestRemoveAllLorasOnClosedContext(t *testing.T) {
	c := &Context{}
	if err := c.RemoveAllLoras(); !errors.Is(err, ErrClosed) {
		t.Fatalf("RemoveAllLoras on a closed context = %v, want ErrClosed", err)
	}
}

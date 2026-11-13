public class MockStudio {
    public Object list;
    public class Positions { public void setPositionList(Object pl) { list = pl; } }
    public class Acq {
        public Object getAcquisitionSettings() { throw new UnsupportedOperationException("mock: no MDA"); }
    }
    public Positions positions() { return new Positions(); }
    public Acq acquisitions() { return new Acq(); }
}

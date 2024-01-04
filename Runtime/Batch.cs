
namespace MachineLearning
{
    public class Batch
    {
        public DataPoint[] data;
        public int size;

        public Batch(int size)
        {
            this.size = size;
            this.data = new DataPoint[size];
        }

        public void addData(DataPoint data, int index)
        {
            this.data[index] = data;
        }

        public void setData(DataPoint[] data)
        {
            this.data = data;
        }
    }
}
use crate::map_fanout_example::{measure, percentile, Shape};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_fanout_example_runs_small() {
    for shape in [Shape::Map, Shape::FanOut] {
        let times = measure(shape, 4, 2).await;
        assert_eq!(times.len(), 2, "{shape:?}");
        assert!(
            percentile(&times, 50) <= percentile(&times, 99),
            "{shape:?}"
        );
    }
}

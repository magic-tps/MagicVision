/// Optional capability for a detector whose remote instance may be asleep.
abstract class InferenceReadinessService {
  Future<void> ensureReady();
}

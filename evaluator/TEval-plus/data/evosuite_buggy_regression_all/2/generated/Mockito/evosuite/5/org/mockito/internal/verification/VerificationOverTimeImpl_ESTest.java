/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:35:10 GMT 2023
 */

package org.mockito.internal.verification;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;
import org.mockito.internal.util.Timer;
import org.mockito.internal.verification.AtMost;
import org.mockito.internal.verification.NoMoreInteractions;
import org.mockito.internal.verification.VerificationOverTimeImpl;
import org.mockito.internal.verification.api.VerificationData;
import org.mockito.verification.Timeout;
import org.mockito.verification.VerificationMode;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class VerificationOverTimeImpl_ESTest extends VerificationOverTimeImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-326L), (-326L), (VerificationMode) null, true);
      long long0 = verificationOverTimeImpl0.getDuration();
      assertEquals((-326L), verificationOverTimeImpl0.getPollingPeriod());
      assertEquals((-326L), long0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Timer timer0 = new Timer(0L);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl(0L, 0L, (VerificationMode) null, false, timer0);
      VerificationMode verificationMode0 = verificationOverTimeImpl0.getDelegate();
      assertNull(verificationMode0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-837L), (-837L), noMoreInteractions0, true);
      long long0 = verificationOverTimeImpl0.getPollingPeriod();
      assertEquals((-837L), long0);
      assertEquals((-837L), verificationOverTimeImpl0.getDuration());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      Timeout timeout0 = new Timeout((-824L), noMoreInteractions0);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-824L), 1L, timeout0, false);
      // Undeclared exception!
      verificationOverTimeImpl0.verify((VerificationData) null);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      Timeout timeout0 = new Timeout((-826L), noMoreInteractions0);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-826L), 1L, timeout0, true);
      verificationOverTimeImpl0.verify((VerificationData) null);
      assertEquals((-826L), verificationOverTimeImpl0.getPollingPeriod());
      assertEquals(1L, verificationOverTimeImpl0.getDuration());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      AtMost atMost0 = new AtMost(4);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl(627L, 627L, atMost0, false);
      boolean boolean0 = verificationOverTimeImpl0.canRecoverFromFailure(atMost0);
      assertFalse(boolean0);
      assertEquals(627L, verificationOverTimeImpl0.getPollingPeriod());
      assertEquals(627L, verificationOverTimeImpl0.getDuration());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl(0L, 0L, noMoreInteractions0, false);
      boolean boolean0 = verificationOverTimeImpl0.canRecoverFromFailure(noMoreInteractions0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl(0L, 0L, (VerificationMode) null, false);
      boolean boolean0 = verificationOverTimeImpl0.canRecoverFromFailure((VerificationMode) null);
      assertTrue(boolean0);
  }
}

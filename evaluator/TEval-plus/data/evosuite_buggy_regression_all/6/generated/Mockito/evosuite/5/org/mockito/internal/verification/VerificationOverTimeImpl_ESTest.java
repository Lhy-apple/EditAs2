/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:58:35 GMT 2023
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
import org.mockito.verification.After;
import org.mockito.verification.Timeout;
import org.mockito.verification.VerificationMode;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class VerificationOverTimeImpl_ESTest extends VerificationOverTimeImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-16L), (-16L), noMoreInteractions0, true);
      long long0 = verificationOverTimeImpl0.getDuration();
      assertEquals((-16L), verificationOverTimeImpl0.getPollingPeriod());
      assertEquals((-16L), long0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      Timer timer0 = new Timer((-2531L));
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-2531L), (-2531L), noMoreInteractions0, false, timer0);
      verificationOverTimeImpl0.getDelegate();
      assertEquals((-2531L), verificationOverTimeImpl0.getPollingPeriod());
      assertEquals((-2531L), verificationOverTimeImpl0.getDuration());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      After after0 = new After(435L, (VerificationMode) null);
      Timer timer0 = new Timer(435L);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl(435L, 435L, after0, true, timer0);
      long long0 = verificationOverTimeImpl0.getPollingPeriod();
      assertEquals(435L, verificationOverTimeImpl0.getDuration());
      assertEquals(435L, long0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      After after0 = new After(0L, (-3209L), noMoreInteractions0);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl(0L, 0L, after0, true);
      verificationOverTimeImpl0.verify((VerificationData) null);
      assertEquals(0L, verificationOverTimeImpl0.getPollingPeriod());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      Timeout timeout0 = new Timeout((-1L), noMoreInteractions0);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-1L), 0L, timeout0, false);
      // Undeclared exception!
      verificationOverTimeImpl0.verify((VerificationData) null);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      Timer timer0 = new Timer(564L);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl(564L, 564L, noMoreInteractions0, false, timer0);
      AtMost atMost0 = new AtMost(417);
      boolean boolean0 = verificationOverTimeImpl0.canRecoverFromFailure(atMost0);
      assertFalse(boolean0);
      assertEquals(564L, verificationOverTimeImpl0.getPollingPeriod());
      assertEquals(564L, verificationOverTimeImpl0.getDuration());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-1L), (-1L), noMoreInteractions0, false);
      boolean boolean0 = verificationOverTimeImpl0.canRecoverFromFailure((VerificationMode) null);
      assertEquals((-1L), verificationOverTimeImpl0.getDuration());
      assertTrue(boolean0);
      assertEquals((-1L), verificationOverTimeImpl0.getPollingPeriod());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-1L), (-1L), noMoreInteractions0, false);
      boolean boolean0 = verificationOverTimeImpl0.canRecoverFromFailure(noMoreInteractions0);
      assertEquals((-1L), verificationOverTimeImpl0.getDuration());
      assertFalse(boolean0);
      assertEquals((-1L), verificationOverTimeImpl0.getPollingPeriod());
  }
}
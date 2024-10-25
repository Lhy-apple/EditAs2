/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:17:17 GMT 2023
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
import org.mockito.verification.VerificationMode;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class VerificationOverTimeImpl_ESTest extends VerificationOverTimeImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl(1302L, 1302L, (VerificationMode) null, true);
      long long0 = verificationOverTimeImpl0.getDuration();
      assertEquals(1302L, long0);
      assertEquals(1302L, verificationOverTimeImpl0.getPollingPeriod());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-2844L), (-2844L), (VerificationMode) null, false);
      verificationOverTimeImpl0.getDelegate();
      assertEquals((-2844L), verificationOverTimeImpl0.getDuration());
      assertEquals((-2844L), verificationOverTimeImpl0.getPollingPeriod());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-2844L), (-2844L), (VerificationMode) null, false);
      long long0 = verificationOverTimeImpl0.getPollingPeriod();
      assertEquals((-2844L), verificationOverTimeImpl0.getDuration());
      assertEquals((-2844L), long0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-2844L), (-2844L), (VerificationMode) null, true);
      VerificationOverTimeImpl verificationOverTimeImpl1 = new VerificationOverTimeImpl((-2844L), 3946L, verificationOverTimeImpl0, true);
      verificationOverTimeImpl1.verify((VerificationData) null);
      assertEquals((-2844L), verificationOverTimeImpl1.getPollingPeriod());
      assertEquals(3946L, verificationOverTimeImpl1.getDuration());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-603L), (-603L), (VerificationMode) null, false);
      VerificationOverTimeImpl verificationOverTimeImpl1 = new VerificationOverTimeImpl((-603L), 1L, verificationOverTimeImpl0, false);
      // Undeclared exception!
      verificationOverTimeImpl1.verify((VerificationData) null);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      AtMost atMost0 = new AtMost(724);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-1L), (-1L), atMost0, false);
      boolean boolean0 = verificationOverTimeImpl0.canRecoverFromFailure(atMost0);
      assertEquals((-1L), verificationOverTimeImpl0.getPollingPeriod());
      assertFalse(boolean0);
      assertEquals((-1L), verificationOverTimeImpl0.getDuration());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl((-1890L), (-1890L), (VerificationMode) null, true);
      boolean boolean0 = verificationOverTimeImpl0.canRecoverFromFailure((VerificationMode) null);
      assertTrue(boolean0);
      assertEquals((-1890L), verificationOverTimeImpl0.getDuration());
      assertEquals((-1890L), verificationOverTimeImpl0.getPollingPeriod());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      Timer timer0 = new Timer(0L);
      VerificationOverTimeImpl verificationOverTimeImpl0 = new VerificationOverTimeImpl(0L, 0L, noMoreInteractions0, false, timer0);
      boolean boolean0 = verificationOverTimeImpl0.canRecoverFromFailure(noMoreInteractions0);
      assertFalse(boolean0);
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:59:46 GMT 2023
 */

package org.jsoup;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.IOException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockIOException;
import org.jsoup.UncheckedIOException;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class UncheckedIOException_ESTest extends UncheckedIOException_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      MockIOException mockIOException0 = new MockIOException("");
      UncheckedIOException uncheckedIOException0 = new UncheckedIOException(mockIOException0);
      IOException iOException0 = uncheckedIOException0.ioException();
      assertSame(mockIOException0, iOException0);
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:38:04 GMT 2023
 */

package org.apache.commons.compress.archivers.sevenz;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.IOException;
import org.apache.commons.compress.archivers.sevenz.SevenZFile;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.testdata.EvoSuiteFile;
import org.evosuite.runtime.testdata.FileSystemHandling;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SevenZFile_ESTest extends SevenZFile_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      EvoSuiteFile evoSuiteFile0 = new EvoSuiteFile("org.tukaani.xz.FinishableOutputStream/org.tukaani.xz.FinishableOutputStream");
      FileSystemHandling.appendStringToFile(evoSuiteFile0, "org.tukaani.xz.FinishableOutputStream");
      MockFile mockFile0 = new MockFile("org.tukaani.xz.FinishableOutputStream", "org.tukaani.xz.FinishableOutputStream");
      SevenZFile sevenZFile0 = null;
      try {
        sevenZFile0 = new SevenZFile(mockFile0);
        fail("Expecting exception: IOException");
      
      } catch(Throwable e) {
         //
         // Unsupported 7z version (107,97)
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.SevenZFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      boolean boolean0 = SevenZFile.matches(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte)55;
      boolean boolean0 = SevenZFile.matches(byteArray0, (byte)55);
      assertFalse(boolean0);
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:25:45 GMT 2023
 */

package org.apache.commons.compress.archivers.dump;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.PipedInputStream;
import java.io.SequenceInputStream;
import org.apache.commons.compress.archivers.dump.DumpArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DumpArchiveInputStream_ESTest extends DumpArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream(1398);
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(pipedInputStream0, pipedInputStream0);
      DumpArchiveInputStream dumpArchiveInputStream0 = null;
      try {
        dumpArchiveInputStream0 = new DumpArchiveInputStream(sequenceInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // Pipe not connected
         //
         verifyException("org.apache.commons.compress.archivers.dump.DumpArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      // Undeclared exception!
      try { 
        DumpArchiveInputStream.matches(byteArray0, (byte)126);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 27
         //
         verifyException("org.apache.commons.compress.archivers.dump.DumpArchiveUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      boolean boolean0 = DumpArchiveInputStream.matches(byteArray0, (-1033));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      // Undeclared exception!
      try { 
        DumpArchiveInputStream.matches(byteArray0, 1119);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 27
         //
         verifyException("org.apache.commons.compress.archivers.dump.DumpArchiveUtil", e);
      }
  }
}

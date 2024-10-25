/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:13:23 GMT 2023
 */

package org.apache.commons.compress.compressors.bzip2;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BZip2CompressorInputStream_ESTest extends BZip2CompressorInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BZip2CompressorInputStream bZip2CompressorInputStream0 = null;
      try {
        bZip2CompressorInputStream0 = new BZip2CompressorInputStream((InputStream) null);
        fail("Expecting exception: IOException");
      
      } catch(Throwable e) {
         //
         // No InputStream
         //
         verifyException("org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-3949), (-3949));
      BZip2CompressorInputStream bZip2CompressorInputStream0 = null;
      try {
        bZip2CompressorInputStream0 = new BZip2CompressorInputStream(byteArrayInputStream0);
        fail("Expecting exception: IOException");
      
      } catch(Throwable e) {
         //
         // Stream is not in the BZip2 format
         //
         verifyException("org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)66;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      BZip2CompressorInputStream bZip2CompressorInputStream0 = null;
      try {
        bZip2CompressorInputStream0 = new BZip2CompressorInputStream(byteArrayInputStream0);
        fail("Expecting exception: IOException");
      
      } catch(Throwable e) {
         //
         // Stream is not in the BZip2 format
         //
         verifyException("org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      boolean boolean0 = BZip2CompressorInputStream.matches(byteArray0, 66);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      boolean boolean0 = BZip2CompressorInputStream.matches(byteArray0, (-1291));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)66;
      boolean boolean0 = BZip2CompressorInputStream.matches(byteArray0, 66);
      assertFalse(boolean0);
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:25:43 GMT 2023
 */

package org.apache.commons.compress.utils;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.util.zip.Adler32;
import java.util.zip.Checksum;
import org.apache.commons.compress.utils.ChecksumCalculatingInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ChecksumCalculatingInputStream_ESTest extends ChecksumCalculatingInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Adler32 adler32_0 = new Adler32();
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(byteArrayInputStream0);
      ChecksumCalculatingInputStream checksumCalculatingInputStream0 = new ChecksumCalculatingInputStream(adler32_0, bufferedInputStream0);
      long long0 = checksumCalculatingInputStream0.skip((-821L));
      assertEquals(65537L, checksumCalculatingInputStream0.getValue());
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Adler32 adler32_0 = new Adler32();
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(byteArrayInputStream0);
      ChecksumCalculatingInputStream checksumCalculatingInputStream0 = new ChecksumCalculatingInputStream(adler32_0, bufferedInputStream0);
      int int0 = checksumCalculatingInputStream0.read(byteArray0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-76), (byte)122);
      ChecksumCalculatingInputStream checksumCalculatingInputStream0 = new ChecksumCalculatingInputStream((Checksum) null, byteArrayInputStream0);
      // Undeclared exception!
      try { 
        checksumCalculatingInputStream0.getValue();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ChecksumCalculatingInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-76), (byte)122);
      byteArrayInputStream0.skip(4805L);
      ChecksumCalculatingInputStream checksumCalculatingInputStream0 = new ChecksumCalculatingInputStream((Checksum) null, byteArrayInputStream0);
      int int0 = checksumCalculatingInputStream0.read(byteArray0);
      assertEquals((-1), int0);
  }
}
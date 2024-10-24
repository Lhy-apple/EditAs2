/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:29:00 GMT 2023
 */

package org.apache.commons.compress.utils;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.util.zip.CRC32;
import java.util.zip.Checksum;
import org.apache.commons.compress.utils.ChecksumCalculatingInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ChecksumCalculatingInputStream_ESTest extends ChecksumCalculatingInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, 0);
      ChecksumCalculatingInputStream checksumCalculatingInputStream0 = new ChecksumCalculatingInputStream((Checksum) null, byteArrayInputStream0);
      int int0 = checksumCalculatingInputStream0.read(byteArray0);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      CRC32 cRC32_0 = new CRC32();
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ChecksumCalculatingInputStream checksumCalculatingInputStream0 = new ChecksumCalculatingInputStream(cRC32_0, byteArrayInputStream0);
      long long0 = checksumCalculatingInputStream0.getValue();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      CRC32 cRC32_0 = new CRC32();
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ChecksumCalculatingInputStream checksumCalculatingInputStream0 = new ChecksumCalculatingInputStream(cRC32_0, byteArrayInputStream0);
      long long0 = checksumCalculatingInputStream0.skip((byte)0);
      assertEquals(3523407757L, checksumCalculatingInputStream0.getValue());
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)80);
      ChecksumCalculatingInputStream checksumCalculatingInputStream0 = new ChecksumCalculatingInputStream((Checksum) null, byteArrayInputStream0);
      // Undeclared exception!
      try { 
        checksumCalculatingInputStream0.read(byteArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ChecksumCalculatingInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, 0);
      ChecksumCalculatingInputStream checksumCalculatingInputStream0 = new ChecksumCalculatingInputStream((Checksum) null, byteArrayInputStream0);
      long long0 = checksumCalculatingInputStream0.skip(0L);
      assertEquals(0L, long0);
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:29:20 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import org.apache.commons.compress.archivers.zip.ZipArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipArchiveInputStream_ESTest extends ZipArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(pipedInputStream0);
      try { 
        zipArchiveInputStream0.getNextEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(byteArrayInputStream0);
      zipArchiveInputStream0.close();
      zipArchiveInputStream0.getNextZipEntry();
      assertEquals(3, byteArrayInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      byte[] byteArray0 = new byte[33];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(byteArrayInputStream0);
      zipArchiveInputStream0.getNextZipEntry();
      assertEquals(3, byteArrayInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      long long0 = zipArchiveInputStream0.skip(648L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      zipArchiveInputStream0.close();
      try { 
        zipArchiveInputStream0.skip(67324752L);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // The stream is closed
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(pipedInputStream0);
      zipArchiveInputStream0.close();
      zipArchiveInputStream0.close();
      assertEquals(0, zipArchiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      // Undeclared exception!
      try { 
        zipArchiveInputStream0.skip((-2478L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(pipedInputStream0);
      long long0 = zipArchiveInputStream0.skip(0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      boolean boolean0 = ZipArchiveInputStream.matches(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)80;
      boolean boolean0 = ZipArchiveInputStream.matches(byteArray0, (byte)80);
      assertFalse(boolean0);
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:52:43 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipArchiveInputStream_ESTest extends ZipArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      try { 
        zipArchiveInputStream0.getNextEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream closed
         //
         verifyException("java.io.PushbackInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      zipArchiveInputStream0.close();
      ZipArchiveEntry zipArchiveEntry0 = zipArchiveInputStream0.getNextZipEntry();
      assertNull(zipArchiveEntry0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      boolean boolean0 = zipArchiveInputStream0.canReadEntryData((ArchiveEntry) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(pipedInputStream0);
      boolean boolean0 = zipArchiveInputStream0.canReadEntryData(zipArchiveEntry0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      ZipArchiveInputStream zipArchiveInputStream1 = new ZipArchiveInputStream(zipArchiveInputStream0);
      zipArchiveInputStream1.getNextZipEntry();
      assertEquals(0L, zipArchiveInputStream1.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      zipArchiveInputStream0.close();
      ZipArchiveInputStream zipArchiveInputStream1 = new ZipArchiveInputStream(zipArchiveInputStream0);
      try { 
        zipArchiveInputStream1.getNextZipEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // The stream is closed
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0);
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(pipedInputStream0);
      zipArchiveInputStream0.close();
      zipArchiveInputStream0.close();
      assertEquals(0, zipArchiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(pipedInputStream0);
      // Undeclared exception!
      try { 
        zipArchiveInputStream0.skip((-2422L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.zip.ZipArchiveInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(byteArrayInputStream0);
      long long0 = zipArchiveInputStream0.skip(0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      long long0 = zipArchiveInputStream0.skip(78L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      long long0 = zipArchiveInputStream0.skip(1777L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[13];
      boolean boolean0 = ZipArchiveInputStream.matches(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte)80;
      boolean boolean0 = ZipArchiveInputStream.matches(byteArray0, (byte)80);
      assertFalse(boolean0);
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:47:01 GMT 2023
 */

package org.apache.commons.compress.archivers.zip;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.arj.ArjArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipArchiveInputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ZipArchiveInputStream_ESTest extends ZipArchiveInputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[46];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(byteArrayInputStream0);
      zipArchiveInputStream0.getNextZipEntry();
      assertEquals(16, byteArrayInputStream0.available());
      assertEquals(30L, zipArchiveInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      zipArchiveInputStream0.close();
      ArchiveEntry archiveEntry0 = zipArchiveInputStream0.getNextEntry();
      assertNull(archiveEntry0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      ArjArchiveEntry arjArchiveEntry0 = new ArjArchiveEntry();
      boolean boolean0 = zipArchiveInputStream0.canReadEntryData(arjArchiveEntry0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      ZipArchiveEntry zipArchiveEntry0 = new ZipArchiveEntry();
      boolean boolean0 = zipArchiveInputStream0.canReadEntryData(zipArchiveEntry0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      long long0 = zipArchiveInputStream0.skip(4294967296L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      zipArchiveInputStream0.close();
      try { 
        zipArchiveInputStream0.read();
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
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      zipArchiveInputStream0.close();
      zipArchiveInputStream0.close();
      assertEquals(0, zipArchiveInputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      // Undeclared exception!
      try { 
        zipArchiveInputStream0.skip((-1));
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
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream((InputStream) null);
      long long0 = zipArchiveInputStream0.skip(0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream((byte)114);
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(pipedInputStream0);
      long long0 = zipArchiveInputStream0.skip((byte)114);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      boolean boolean0 = ZipArchiveInputStream.matches(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte)80;
      boolean boolean0 = ZipArchiveInputStream.matches(byteArray0, (byte)80);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[15];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ZipArchiveInputStream zipArchiveInputStream0 = new ZipArchiveInputStream(byteArrayInputStream0);
      zipArchiveInputStream0.getNextZipEntry();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(15, zipArchiveInputStream0.getCount());
  }
}
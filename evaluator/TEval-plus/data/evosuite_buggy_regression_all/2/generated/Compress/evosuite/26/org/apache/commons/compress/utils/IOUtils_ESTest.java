/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:46:00 GMT 2023
 */

package org.apache.commons.compress.utils;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import org.apache.commons.compress.utils.IOUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class IOUtils_ESTest extends IOUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream(1);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("org.apache.commons.compress.utils.IOUtils", false);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(mockFileOutputStream0, 8024);
      // Undeclared exception!
      IOUtils.copy((InputStream) pipedInputStream0, (OutputStream) bufferedOutputStream0, 0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-62), (byte) (-2));
      long long0 = IOUtils.skip(byteArrayInputStream0, (byte) (-62));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-1), (-1));
      long long0 = IOUtils.skip(byteArrayInputStream0, (byte)1);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-1), (-1));
      // Undeclared exception!
      try { 
        IOUtils.readFully((InputStream) byteArrayInputStream0, byteArray0, 0, (-3831));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.IOUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0);
      // Undeclared exception!
      try { 
        IOUtils.readFully((InputStream) pipedInputStream0, byteArray0, (int) (byte) (-1), 4096);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.IOUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 3804, (byte) (-47));
      byte[] byteArray1 = IOUtils.toByteArray(byteArrayInputStream0);
      // Undeclared exception!
      try { 
        IOUtils.readFully((InputStream) byteArrayInputStream0, byteArray1, 1, (int) (byte)1);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.IOUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)1);
      int int0 = IOUtils.readFully((InputStream) byteArrayInputStream0, byteArray0, 0, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 1, 1);
      int int0 = IOUtils.readFully((InputStream) byteArrayInputStream0, byteArray0);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      IOUtils.closeQuietly((Closeable) null);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(0);
      IOUtils.closeQuietly(byteArrayOutputStream0);
      assertEquals(0, byteArrayOutputStream0.size());
  }
}

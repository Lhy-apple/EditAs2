/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:51:25 GMT 2023
 */

package org.apache.commons.compress.compressors;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import org.apache.commons.compress.compressors.CompressorInputStream;
import org.apache.commons.compress.compressors.CompressorOutputStream;
import org.apache.commons.compress.compressors.CompressorStreamFactory;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CompressorStreamFactory_ESTest extends CompressorStreamFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      boolean boolean0 = compressorStreamFactory0.getDecompressConcatenated();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(false);
      // Undeclared exception!
      try { 
        compressorStreamFactory0.setDecompressConcatenated(false);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Cannot override the setting defined by the constructor
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      compressorStreamFactory0.setDecompressConcatenated(true);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)59, (byte)59);
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      try { 
        compressorStreamFactory0.createCompressorInputStream((InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // No Compressor found for the stream signature.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorInputStream((InputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Stream must not be null.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorInputStream((InputStream) pipedInputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Mark is not supported.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null);
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorInputStream((String) null, (InputStream) bufferedInputStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Compressor name and stream must not be null.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-71), (byte) (-71));
      CompressorInputStream compressorInputStream0 = compressorStreamFactory0.createCompressorInputStream("deflate", (InputStream) byteArrayInputStream0);
      assertEquals(0L, compressorInputStream0.getBytesRead());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorInputStream("deflate", (InputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Compressor name and stream must not be null.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-1740), (-1740));
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      try { 
        compressorStreamFactory0.createCompressorInputStream("gz", (InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not create CompressorInputStream.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte) (-71), (byte) (-71));
      try { 
        compressorStreamFactory0.createCompressorInputStream("bzip2", (InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not create CompressorInputStream.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 1995, 1995);
      try { 
        compressorStreamFactory0.createCompressorInputStream("xz", (InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not create CompressorInputStream.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null);
      try { 
        compressorStreamFactory0.createCompressorInputStream("lzma", (InputStream) bufferedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not create CompressorInputStream.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-1740), (-1740));
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      try { 
        compressorStreamFactory0.createCompressorInputStream("pack200", (InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not create CompressorInputStream.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null);
      try { 
        compressorStreamFactory0.createCompressorInputStream("snappy-raw", (InputStream) bufferedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not create CompressorInputStream.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null);
      try { 
        compressorStreamFactory0.createCompressorInputStream("snappy-framed", (InputStream) bufferedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not create CompressorInputStream.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null);
      try { 
        compressorStreamFactory0.createCompressorInputStream("z", (InputStream) bufferedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not create CompressorInputStream.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream((InputStream) null);
      try { 
        compressorStreamFactory0.createCompressorInputStream("7z", (InputStream) bufferedInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Compressor: 7z not found.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      MockFile mockFile0 = new MockFile((File) null, "snappy-raw");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorOutputStream((String) null, mockPrintStream0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Compressor name and stream must not be null.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CompressorOutputStream compressorOutputStream0 = compressorStreamFactory0.createCompressorOutputStream("gz", byteArrayOutputStream0);
      assertNotNull(compressorOutputStream0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorOutputStream("", (OutputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Compressor name and stream must not be null.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      try { 
        compressorStreamFactory0.createCompressorOutputStream("bzip2", pipedOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Could not create CompressorOutputStream
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      try { 
        compressorStreamFactory0.createCompressorOutputStream("wX-F", pipedOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Compressor: wX-F not found.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream((OutputStream) null);
      CompressorOutputStream compressorOutputStream0 = compressorStreamFactory0.createCompressorOutputStream("xz", bufferedOutputStream0);
      assertNotNull(compressorOutputStream0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      MockFile mockFile0 = new MockFile("gz", "xz");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      compressorStreamFactory0.createCompressorOutputStream("pack200", mockPrintStream0);
      assertEquals(0L, mockFile0.length());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      MockPrintStream mockPrintStream0 = new MockPrintStream("O]jR");
      CompressorOutputStream compressorOutputStream0 = compressorStreamFactory0.createCompressorOutputStream("deflate", mockPrintStream0);
      assertNotNull(compressorOutputStream0);
  }
}
/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:25:15 GMT 2023
 */

package org.apache.commons.compress.compressors;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import org.apache.commons.compress.compressors.CompressorOutputStream;
import org.apache.commons.compress.compressors.CompressorStreamFactory;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CompressorStreamFactory_ESTest extends CompressorStreamFactory_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      boolean boolean0 = compressorStreamFactory0.getDecompressConcatenated();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      compressorStreamFactory0.setDecompressConcatenated(true);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      // Undeclared exception!
      try { 
        compressorStreamFactory0.setDecompressConcatenated(true);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Cannot override the setting defined by the constructor
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[26];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
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
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(false);
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorInputStream((String) null, (InputStream) byteArrayInputStream0);
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
      byte[] byteArray0 = new byte[26];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      compressorStreamFactory0.createCompressorInputStream("deflate", (InputStream) byteArrayInputStream0);
      assertEquals(26, byteArrayInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorInputStream("", (InputStream) null);
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
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
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
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
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
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
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
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      try { 
        compressorStreamFactory0.createCompressorInputStream("lzma", (InputStream) pipedInputStream0);
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
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        compressorStreamFactory0.createCompressorInputStream("pack200", (InputStream) pipedInputStream0);
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
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      try { 
        compressorStreamFactory0.createCompressorInputStream("snappy-raw", (InputStream) pipedInputStream0);
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
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(false);
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      try { 
        compressorStreamFactory0.createCompressorInputStream("snappy-framed", (InputStream) byteArrayInputStream0);
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
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      try { 
        compressorStreamFactory0.createCompressorInputStream("z", (InputStream) byteArrayInputStream0);
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
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      try { 
        compressorStreamFactory0.createCompressorInputStream("e]:x=b8 k4fh( o5", (InputStream) byteArrayInputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Compressor: e]:x=b8 k4fh( o5 not found.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream(pipedInputStream0);
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorOutputStream((String) null, pipedOutputStream0);
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
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      MockPrintStream mockPrintStream0 = new MockPrintStream("compressor name and stream must not be null.");
      CompressorOutputStream compressorOutputStream0 = compressorStreamFactory0.createCompressorOutputStream("xz", mockPrintStream0);
      assertNotNull(compressorOutputStream0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      // Undeclared exception!
      try { 
        compressorStreamFactory0.createCompressorOutputStream("lzma ictionary is too big for this implementation", (OutputStream) null);
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
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      try { 
        compressorStreamFactory0.createCompressorOutputStream("gz", pipedOutputStream0);
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
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream(pipedInputStream0);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(pipedOutputStream0);
      BZip2CompressorOutputStream bZip2CompressorOutputStream0 = (BZip2CompressorOutputStream)compressorStreamFactory0.createCompressorOutputStream("bzip2", bufferedOutputStream0);
      assertEquals(9, BZip2CompressorOutputStream.MAX_BLOCKSIZE);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      try { 
        compressorStreamFactory0.createCompressorOutputStream("unskippable chunk with type ", byteArrayOutputStream0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Compressor: unskippable chunk with type  not found.
         //
         verifyException("org.apache.commons.compress.compressors.CompressorStreamFactory", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream(pipedInputStream0);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(pipedOutputStream0);
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      CompressorOutputStream compressorOutputStream0 = compressorStreamFactory0.createCompressorOutputStream("pack200", bufferedOutputStream0);
      assertNotNull(compressorOutputStream0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      CompressorStreamFactory compressorStreamFactory0 = new CompressorStreamFactory(true);
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream(pipedInputStream0);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(pipedOutputStream0);
      CompressorOutputStream compressorOutputStream0 = compressorStreamFactory0.createCompressorOutputStream("deflate", bufferedOutputStream0);
      assertNotNull(compressorOutputStream0);
  }
}

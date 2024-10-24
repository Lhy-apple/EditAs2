/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:35:50 GMT 2023
 */

package org.apache.commons.compress.archivers.sevenz;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PipedOutputStream;
import java.io.PushbackInputStream;
import java.io.SequenceInputStream;
import java.util.Enumeration;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.sevenz.Coder;
import org.apache.commons.compress.archivers.sevenz.Coders;
import org.apache.commons.compress.archivers.sevenz.SevenZMethod;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Coders_ESTest extends Coders_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Coders coders0 = new Coders();
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Coders.BZIP2Decoder coders_BZIP2Decoder0 = new Coders.BZIP2Decoder();
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)2, (byte) (-33));
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(byteArrayInputStream0);
      Coder coder0 = new Coder();
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode(pushbackInputStream0, coder0, byteArray0);
      // Undeclared exception!
      try { 
        coders_BZIP2Decoder0.decode(inputStream0, coder0, byteArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders$AES256SHA256Decoder$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Coders.LZMADecoder coders_LZMADecoder0 = new Coders.LZMADecoder();
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        coders_LZMADecoder0.encode(pipedOutputStream0, byteArray0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // method doesn't support writing
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders$CoderBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0);
      SevenZMethod sevenZMethod0 = SevenZMethod.BZIP2;
      OutputStream outputStream0 = Coders.addEncoder(mockPrintStream0, sevenZMethod0, (byte[]) null);
      SevenZMethod sevenZMethod1 = SevenZMethod.COPY;
      BZip2CompressorOutputStream bZip2CompressorOutputStream0 = (BZip2CompressorOutputStream)Coders.addEncoder(outputStream0, sevenZMethod1, (byte[]) null);
      assertEquals(9, BZip2CompressorOutputStream.MAX_BLOCKSIZE);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      SevenZMethod sevenZMethod0 = SevenZMethod.DEFLATE;
      OutputStream outputStream0 = Coders.addEncoder(pipedOutputStream0, sevenZMethod0, (byte[]) null);
      assertNotNull(outputStream0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      byte[] byteArray0 = new byte[16];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coder coder0 = new Coder();
      Coders.CopyDecoder coders_CopyDecoder0 = new Coders.CopyDecoder();
      InputStream inputStream0 = coders_CopyDecoder0.decode(byteArrayInputStream0, coder0, byteArray0);
      assertEquals(16, inputStream0.available());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      byte[] byteArray0 = new byte[16];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coder coder0 = new Coder();
      try { 
        Coders.addDecoder(byteArrayInputStream0, coder0, byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unsupported compression method null
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Coders.CoderId[] coders_CoderIdArray0 = new Coders.CoderId[6];
      SevenZMethod sevenZMethod0 = SevenZMethod.LZMA;
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      Coders.CoderId coders_CoderId0 = new Coders.CoderId(sevenZMethod0, coders_AES256SHA256Decoder0);
      coders_CoderIdArray0[0] = coders_CoderId0;
      coders_CoderIdArray0[1] = coders_CoderIdArray0[0];
      coders_CoderIdArray0[2] = coders_CoderId0;
      coders_CoderIdArray0[3] = coders_CoderIdArray0[0];
      coders_CoderIdArray0[4] = coders_CoderIdArray0[2];
      coders_CoderIdArray0[5] = coders_CoderId0;
      Coders.coderTable = coders_CoderIdArray0;
      SevenZMethod sevenZMethod1 = SevenZMethod.LZMA2;
      byte[] byteArray0 = new byte[4];
      try { 
        Coders.addEncoder((OutputStream) null, sevenZMethod1, byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unsupported compression method LZMA2
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Coders.BZIP2Decoder coders_BZIP2Decoder0 = new Coders.BZIP2Decoder();
      Coders.DeflateDecoder coders_DeflateDecoder0 = new Coders.DeflateDecoder();
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coder coder0 = new Coder();
      InputStream inputStream0 = coders_DeflateDecoder0.decode(byteArrayInputStream0, coder0, byteArray0);
      try { 
        coders_BZIP2Decoder0.decode(inputStream0, coder0, byteArray0);
        fail("Expecting exception: ZipException");
      
      } catch(ZipException e) {
         //
         // invalid stored block lengths
         //
         verifyException("java.util.zip.InflaterInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Coders.BZIP2Decoder coders_BZIP2Decoder0 = new Coders.BZIP2Decoder();
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      Coder coder0 = new Coder();
      Coders.DeflateDecoder coders_DeflateDecoder0 = new Coders.DeflateDecoder();
      InputStream inputStream0 = coders_DeflateDecoder0.decode(sequenceInputStream0, coder0, (byte[]) null);
      try { 
        coders_BZIP2Decoder0.decode(inputStream0, coder0, (byte[]) null);
        fail("Expecting exception: EOFException");
      
      } catch(EOFException e) {
         //
         // Unexpected end of ZLIB input stream
         //
         verifyException("java.util.zip.InflaterInputStream", e);
      }
  }
}

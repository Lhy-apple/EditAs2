/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:46:14 GMT 2023
 */

package com.fasterxml.jackson.core.base;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.base.ParserMinimalBase;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.json.UTF8StreamJsonParser;
import com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.SequenceInputStream;
import java.io.StringReader;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ParserMinimalBase_ESTest extends ParserMinimalBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      String string0 = ParserMinimalBase._ascii(byteArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("-");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 1715, stringReader0, objectCodec0, charsToNameCanonicalizer0);
      try { 
        readerBasedJsonParser0.nextValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unexpected end-of-inputNo digit following minus sign
         //  at [Source: com.fasterxml.jackson.core.util.BufferRecycler@0000000001; line: 1, column: 3]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("3QoDSp");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-215), stringReader0, objectCodec0, charsToNameCanonicalizer0);
      try { 
        readerBasedJsonParser0.nextValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unexpected character ('Q' (code 81)): Expected space separating root-level values
         //  at [Source: com.fasterxml.jackson.core.util.BufferRecycler@0000000002; line: 1, column: 3]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader(" PY}");
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[8];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-1787), stringReader0, objectCodec0, charsToNameCanonicalizer0, charArray0, (-280), 1, true);
      readerBasedJsonParser0.getLastClearedToken();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      IOContext iOContext0 = new IOContext(bufferRecycler0, byteArrayInputStream0, false);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      UTF8StreamJsonParser uTF8StreamJsonParser0 = new UTF8StreamJsonParser(iOContext0, 2400, byteArrayInputStream0, (ObjectCodec) null, byteQuadsCanonicalizer0, byteArray0, 92, (byte)8, true);
      try { 
        uTF8StreamJsonParser0.nextToken();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Illegal character ((CTRL-CHAR, code 0)): only regular white space (\\r, \\n, \\t) is allowed between tokens
         //  at [Source: java.io.ByteArrayInputStream@0000000003; line: 1, column: -82]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      PipedInputStream pipedInputStream0 = new PipedInputStream(1023);
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(pipedInputStream0, pipedInputStream0);
      DataInputStream dataInputStream0 = new DataInputStream(sequenceInputStream0);
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byte[] byteArray0 = new byte[9];
      UTF8StreamJsonParser uTF8StreamJsonParser0 = new UTF8StreamJsonParser(iOContext0, (-521), dataInputStream0, objectCodec0, byteQuadsCanonicalizer0, byteArray0, (byte)0, 0, false);
      uTF8StreamJsonParser0.getCurrentToken();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byte[] byteArray0 = new byte[0];
      UTF8StreamJsonParser uTF8StreamJsonParser0 = new UTF8StreamJsonParser(iOContext0, 3, (InputStream) null, (ObjectCodec) null, byteQuadsCanonicalizer0, byteArray0, 2, 0, false);
      int int0 = uTF8StreamJsonParser0.getCurrentTokenId();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 33, 0, false);
      boolean boolean0 = readerBasedJsonParser0.hasCurrentToken();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 1, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      boolean boolean0 = readerBasedJsonParser0.hasTokenId(0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 58, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      boolean boolean0 = readerBasedJsonParser0.hasTokenId(2);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-280), 12);
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      UTF8StreamJsonParser uTF8StreamJsonParser0 = new UTF8StreamJsonParser(iOContext0, 3, byteArrayInputStream0, objectCodec0, byteQuadsCanonicalizer0, byteArray0, 1206, 0, true);
      JsonToken jsonToken0 = JsonToken.VALUE_FALSE;
      boolean boolean0 = uTF8StreamJsonParser0.hasToken(jsonToken0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, "(CTRL-CHAR, code 0)", true);
      StringReader stringReader0 = new StringReader("");
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, stringReader0, objectCodec0, charsToNameCanonicalizer0);
      boolean boolean0 = readerBasedJsonParser0.hasToken((JsonToken) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 33, 0, false);
      boolean boolean0 = readerBasedJsonParser0.isExpectedStartArrayToken();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (-280), 12);
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      UTF8StreamJsonParser uTF8StreamJsonParser0 = new UTF8StreamJsonParser(iOContext0, 3, byteArrayInputStream0, objectCodec0, byteQuadsCanonicalizer0, byteArray0, 1206, 0, true);
      boolean boolean0 = uTF8StreamJsonParser0.isExpectedStartObjectToken();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, "(CTRL-CHAR, code 0)", true);
      StringReader stringReader0 = new StringReader("");
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, stringReader0, objectCodec0, charsToNameCanonicalizer0);
      readerBasedJsonParser0.nextValue();
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 33, 0, false);
      JsonParser jsonParser0 = readerBasedJsonParser0.skipChildren();
      assertFalse(jsonParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 33, 0, false);
      readerBasedJsonParser0.clearCurrentToken();
      assertFalse(readerBasedJsonParser0.hasTextCharacters());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("");
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 4161, stringReader0, objectCodec0, charsToNameCanonicalizer0);
      boolean boolean0 = readerBasedJsonParser0.getValueAsBoolean(true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 1, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      int int0 = readerBasedJsonParser0.getValueAsInt();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 1, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      long long0 = readerBasedJsonParser0.getValueAsLong();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 33, 0, false);
      double double0 = readerBasedJsonParser0.getValueAsDouble((double) 3);
      assertEquals(3.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.base.ParserMinimalBase");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 33, 0, false);
      String string0 = readerBasedJsonParser0.getValueAsString();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ObjectCodec objectCodec0 = mock(ObjectCodec.class, new ViolatedAssumptionAnswer());
      char[] charArray0 = new char[17];
      charArray0[2] = '\'';
      charArray0[3] = 'u';
      charArray0[4] = '\\';
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-419), stringReader0, objectCodec0, charsToNameCanonicalizer0, charArray0, 2, 2605, true);
      try { 
        readerBasedJsonParser0.nextValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Illegal unquoted character ((CTRL-CHAR, code 0)): has to be escaped using backslash to be included in string value
         //  at [Source: com.fasterxml.jackson.core.util.BufferRecycler@0000000004; line: 1, column: 8]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      String string0 = ParserMinimalBase._getCharDesc(370);
      assertEquals("'\u0172' (code 370 / 0x172)", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = ParserMinimalBase._asciiBytes("g%q8w{D9F.L<T");
      assertEquals(13, byteArray0.length);
  }
}

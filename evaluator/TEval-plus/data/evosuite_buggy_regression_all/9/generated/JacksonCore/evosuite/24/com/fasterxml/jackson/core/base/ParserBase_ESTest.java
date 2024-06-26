/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:36:42 GMT 2023
 */

package com.fasterxml.jackson.core.base;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.base.ParserBase;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.json.UTF8StreamJsonParser;
import com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import java.io.IOException;
import java.io.PipedInputStream;
import java.io.StringReader;
import java.nio.CharBuffer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ParserBase_ESTest extends ParserBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("<_<TF[1");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-1241), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.getParsingContext();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("<_<TF[1");
      char[] charArray0 = new char[6];
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 0, 2, false);
      readerBasedJsonParser0.isClosed();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("Dr8,?m]CxWZ");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, (char[]) null, 3, 3, false);
      int int0 = readerBasedJsonParser0.getTokenLineNr();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("@8");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.getCurrentValue();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("<_<TF[1");
      char[] charArray0 = new char[6];
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 0, 2, false);
      long long0 = readerBasedJsonParser0.getTokenCharacterOffset();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("<_<TF[1");
      char[] charArray0 = new char[6];
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      stringReader0.read(charBuffer0);
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.nextFieldName();
      int int0 = readerBasedJsonParser0.getIntValue();
      assertEquals(7, readerBasedJsonParser0.getCurrentTokenId());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("<_<TF[1");
      char[] charArray0 = new char[6];
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 0, 2, false);
      readerBasedJsonParser0.version();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 1, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.setCurrentValue(bufferRecycler0);
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, "Zn,XVNdG7Dn$", true);
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byte[] byteArray0 = new byte[18];
      UTF8StreamJsonParser uTF8StreamJsonParser0 = new UTF8StreamJsonParser(iOContext0, 2, pipedInputStream0, (ObjectCodec) null, byteQuadsCanonicalizer0, byteArray0, 0, 2, true);
      try { 
        uTF8StreamJsonParser0.getFloatValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (null) not numeric, can not use numeric value accessors
         //  at [Source: UNKNOWN; line: 1, column: 1]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("Dr8,?m]CxWZ");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, (char[]) null, 3, 3, true);
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.ALLOW_SINGLE_QUOTES;
      readerBasedJsonParser0.configure(jsonParser_Feature0, true);
      assertEquals(19, readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("M>dQ=IL>@u6x");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[3];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 1096, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 18, 751, false);
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.STRICT_DUPLICATE_DETECTION;
      readerBasedJsonParser0.configure(jsonParser_Feature0, true);
      assertEquals(3144, readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("<_<TF[1");
      char[] charArray0 = new char[6];
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 0, 2, false);
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.ALLOW_NON_NUMERIC_NUMBERS;
      ReaderBasedJsonParser readerBasedJsonParser1 = (ReaderBasedJsonParser)readerBasedJsonParser0.configure(jsonParser_Feature0, false);
      assertEquals(3, readerBasedJsonParser1.getFeatureMask());
      assertEquals(1, readerBasedJsonParser1.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      StringReader stringReader0 = new StringReader("Illegal white space character (code 0x%s) as character #%d of 4-char base64 unit: can only used between units");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      IOContext iOContext0 = new IOContext(bufferRecycler0, (Object) null, true);
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, (char[]) null, 104, (-461), true);
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.STRICT_DUPLICATE_DETECTION;
      ReaderBasedJsonParser readerBasedJsonParser1 = (ReaderBasedJsonParser)readerBasedJsonParser0.configure(jsonParser_Feature0, false);
      assertEquals(2, readerBasedJsonParser1.getFeatureMask());
      assertEquals(1, readerBasedJsonParser1.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("JZl9Seq('Uut(");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParser jsonParser0 = readerBasedJsonParser0.setFeatureMask(33);
      assertEquals(33, jsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-564), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.setFeatureMask(0);
      assertEquals(0, readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("Dr8,?m]CxWZ");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 114, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParser jsonParser0 = readerBasedJsonParser0.overrideStdFeatures(114, 2);
      assertEquals(114, jsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.overrideStdFeatures((-1241), 9999);
      assertEquals(8967, readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("<_<TF[1");
      char[] charArray0 = new char[6];
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 0, 2, false);
      readerBasedJsonParser0.setFeatureMask((-1274));
      assertEquals((-1274), readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, "int", false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("int");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 39, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.currentName();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-564), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.overrideCurrentName("");
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.nextToken();
      readerBasedJsonParser0.nextTextValue();
      assertTrue(readerBasedJsonParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      boolean boolean0 = readerBasedJsonParser0.hasTextCharacters();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-564), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      int int0 = readerBasedJsonParser0.getTokenColumnNr();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("JZl9Seq('Uut(");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0._getByteArrayBuilder();
      readerBasedJsonParser0._getByteArrayBuilder();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("Dr8,?m]CxWZ");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-3734), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, (char[]) null, 3, 0, false);
      boolean boolean0 = readerBasedJsonParser0.isNaN();
      assertFalse(boolean0);
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("JZl9Seq('Uut(");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      try { 
        readerBasedJsonParser0.getNumberValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (null) not numeric, can not use numeric value accessors
         //  at [Source: UNKNOWN; line: 1, column: 1]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("");
      char[] charArray0 = new char[1];
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 4, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 2, (-1674), false);
      try { 
        readerBasedJsonParser0.getNumberType();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (null) not numeric, can not use numeric value accessors
         //  at [Source: UNKNOWN; line: 1, column: 3]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-38), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      try { 
        readerBasedJsonParser0.getIntValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (null) not numeric, can not use numeric value accessors
         //  at [Source: (com.fasterxml.jackson.core.util.BufferRecycler); line: 1, column: 1]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, "int", false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("int");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 7005, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      try { 
        readerBasedJsonParser0.getLongValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (null) not numeric, can not use numeric value accessors
         //  at [Source: UNKNOWN; line: 1, column: 1]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("Dr8,?m]CxWZ");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-3734), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, (char[]) null, 3, 0, false);
      try { 
        readerBasedJsonParser0.getBigIntegerValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (null) not numeric, can not use numeric value accessors
         //  at [Source: (com.fasterxml.jackson.core.util.BufferRecycler); line: 1, column: 4]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, "int", false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("int");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 39, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      try { 
        readerBasedJsonParser0.getDecimalValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (null) not numeric, can not use numeric value accessors
         //  at [Source: UNKNOWN; line: 1, column: 1]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      int[] intArray0 = new int[0];
      int[] intArray1 = ParserBase.growArrayBy(intArray0, 3);
      assertEquals(3, intArray1.length);
  }
}

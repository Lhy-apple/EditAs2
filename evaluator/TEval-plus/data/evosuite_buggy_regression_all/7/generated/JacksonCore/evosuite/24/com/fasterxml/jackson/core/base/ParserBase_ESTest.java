/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:49:25 GMT 2023
 */

package com.fasterxml.jackson.core.base;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.base.ParserBase;
import com.fasterxml.jackson.core.filter.FilteringParserDelegate;
import com.fasterxml.jackson.core.filter.TokenFilter;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.json.UTF8StreamJsonParser;
import com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ParserBase_ESTest extends ParserBase_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("vrt{>H=u(}JD;7iS");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, (char[]) null, 33, 1, true);
      readerBasedJsonParser0.getParsingContext();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[2];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 98, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 0, 1321, true);
      readerBasedJsonParser0.isClosed();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("[#%");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      IOContext iOContext0 = new IOContext(bufferRecycler0, byteQuadsCanonicalizer0, true);
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      int int0 = readerBasedJsonParser0.getTokenLineNr();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[2];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 1, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 3978, 48, false);
      readerBasedJsonParser0.getCurrentValue();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      StringReader stringReader0 = new StringReader("");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 687, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      long long0 = readerBasedJsonParser0.getTokenCharacterOffset();
      assertEquals(0L, long0);
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("Illegal unquoted character (");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[5];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 185, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 33, 33, true);
      readerBasedJsonParser0.version();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.setCurrentValue((Object) null);
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 3, 0, true);
      try { 
        readerBasedJsonParser0.getFloatValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (null) not numeric, can not use numeric value accessors
         //  at [Source: UNKNOWN; line: 1, column: 4]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 3, 0, true);
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.ALLOW_NUMERIC_LEADING_ZEROS;
      readerBasedJsonParser0.configure(jsonParser_Feature0, true);
      assertEquals(131, readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("com.fasterxml.jackson.core.PrettyPrinter");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2205, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.STRICT_DUPLICATE_DETECTION;
      ReaderBasedJsonParser readerBasedJsonParser1 = (ReaderBasedJsonParser)readerBasedJsonParser0.enable(jsonParser_Feature0);
      assertEquals(1, readerBasedJsonParser1.getTokenLineNr());
      assertEquals(2205, readerBasedJsonParser1.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("vrt{>H=u(}JD;7iS");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.STRICT_DUPLICATE_DETECTION;
      readerBasedJsonParser0.enable(jsonParser_Feature0);
      assertEquals(2050, readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.ALLOW_UNQUOTED_CONTROL_CHARS;
      ReaderBasedJsonParser readerBasedJsonParser1 = (ReaderBasedJsonParser)readerBasedJsonParser0.disable(jsonParser_Feature0);
      assertEquals(1, readerBasedJsonParser1.getTokenLineNr());
      assertEquals(2, readerBasedJsonParser1.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParser.Feature jsonParser_Feature0 = JsonParser.Feature.STRICT_DUPLICATE_DETECTION;
      ReaderBasedJsonParser readerBasedJsonParser1 = (ReaderBasedJsonParser)readerBasedJsonParser0.disable(jsonParser_Feature0);
      assertEquals(1, readerBasedJsonParser1.getTokenLineNr());
      assertEquals(0, readerBasedJsonParser1.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 1, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParser jsonParser0 = readerBasedJsonParser0.setFeatureMask(1);
      assertEquals(1, jsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 3, 0, true);
      readerBasedJsonParser0.setFeatureMask(33);
      assertEquals(33, readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("Q7f9T$y!4!3gL");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-4558), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParser jsonParser0 = readerBasedJsonParser0.overrideStdFeatures(232, 1);
      assertEquals((-4558), jsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("vrt{>H=u(}JD;7iS");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 33, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0, (char[]) null, 33, 1, false);
      readerBasedJsonParser0.overrideStdFeatures(2058, (-4288));
      assertEquals(2081, readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-4116), (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.setFeatureMask(0);
      assertEquals(0, readerBasedJsonParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[0];
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 3, 0, true);
      readerBasedJsonParser0.getCurrentName();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0.overrideCurrentName("Malformed numeric value (");
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      UTF8StreamJsonParser uTF8StreamJsonParser0 = new UTF8StreamJsonParser(iOContext0, 3, byteArrayInputStream0, (ObjectCodec) null, byteQuadsCanonicalizer0, byteArray0, 3, 0, false);
      uTF8StreamJsonParser0.close();
      uTF8StreamJsonParser0.nextToken();
      assertTrue(uTF8StreamJsonParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      boolean boolean0 = readerBasedJsonParser0.hasTextCharacters();
      assertFalse(boolean0);
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      UTF8StreamJsonParser uTF8StreamJsonParser0 = new UTF8StreamJsonParser(iOContext0, 3, byteArrayInputStream0, (ObjectCodec) null, byteQuadsCanonicalizer0, byteArray0, 3, 0, true);
      int int0 = uTF8StreamJsonParser0.getTokenColumnNr();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("Q7f9T$y!4!3gL");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-4558), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      readerBasedJsonParser0._getByteArrayBuilder();
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      boolean boolean0 = readerBasedJsonParser0.isNaN();
      assertFalse(boolean0);
      assertEquals(1, readerBasedJsonParser0.getTokenLineNr());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
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
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      try { 
        readerBasedJsonParser0.getNumberType();
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
  public void test27()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      byte[] byteArray0 = new byte[0];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      UTF8StreamJsonParser uTF8StreamJsonParser0 = new UTF8StreamJsonParser(iOContext0, 3, byteArrayInputStream0, (ObjectCodec) null, byteQuadsCanonicalizer0, byteArray0, 3, 0, false);
      try { 
        uTF8StreamJsonParser0.getIntValue();
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
  public void test28()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      StringReader stringReader0 = new StringReader("Q7f9T$y!4!3gL");
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, (-4558), stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      TokenFilter tokenFilter0 = TokenFilter.INCLUDE_ALL;
      FilteringParserDelegate filteringParserDelegate0 = new FilteringParserDelegate(readerBasedJsonParser0, tokenFilter0, true, true);
      try { 
        filteringParserDelegate0.getLongValue();
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
  public void test29()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      try { 
        readerBasedJsonParser0.getBigIntegerValue();
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
  public void test30()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
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
      int[] intArray0 = new int[9];
      int[] intArray1 = ParserBase.growArrayBy(intArray0, 1629);
      assertEquals(1638, intArray1.length);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      int[] intArray0 = ParserBase.growArrayBy((int[]) null, 1421);
      assertEquals(1421, intArray0.length);
  }
}

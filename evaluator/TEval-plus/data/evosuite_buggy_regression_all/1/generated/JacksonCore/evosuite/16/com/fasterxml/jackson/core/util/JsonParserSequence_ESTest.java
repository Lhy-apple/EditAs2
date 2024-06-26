/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:30:43 GMT 2023
 */

package com.fasterxml.jackson.core.util;

import org.junit.Test;
import static org.junit.Assert.*;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.io.IOContext;
import com.fasterxml.jackson.core.json.ReaderBasedJsonParser;
import com.fasterxml.jackson.core.sym.CharsToNameCanonicalizer;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.JsonParserSequence;
import java.io.Reader;
import java.io.StringReader;
import java.util.LinkedList;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JsonParserSequence_ESTest extends JsonParserSequence_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 0, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(readerBasedJsonParser0, readerBasedJsonParser0);
      jsonParserSequence0.nextToken();
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      JsonParser[] jsonParserArray0 = new JsonParser[1];
      JsonParserSequence jsonParserSequence0 = new JsonParserSequence(jsonParserArray0);
      int int0 = jsonParserSequence0.containedParsersCount();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      JsonParser[] jsonParserArray0 = new JsonParser[17];
      JsonParserSequence jsonParserSequence0 = new JsonParserSequence(jsonParserArray0);
      JsonParserSequence jsonParserSequence1 = JsonParserSequence.createFlattened(jsonParserSequence0, (JsonParser) null);
      assertEquals(18, jsonParserSequence1.containedParsersCount());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened((JsonParser) null, (JsonParser) null);
      JsonParserSequence jsonParserSequence1 = JsonParserSequence.createFlattened((JsonParser) null, jsonParserSequence0);
      assertEquals(3, jsonParserSequence1.containedParsersCount());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened((JsonParser) null, (JsonParser) null);
      LinkedList<JsonParser> linkedList0 = new LinkedList<JsonParser>();
      JsonParser[] jsonParserArray0 = new JsonParser[5];
      jsonParserArray0[4] = (JsonParser) jsonParserSequence0;
      JsonParserSequence jsonParserSequence1 = new JsonParserSequence(jsonParserArray0);
      jsonParserSequence1.addFlattenedActiveParsers(linkedList0);
      assertEquals(6, linkedList0.size());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 3, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(readerBasedJsonParser0, readerBasedJsonParser0);
      jsonParserSequence0.close();
      assertEquals(3, jsonParserSequence0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, false);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      StringReader stringReader0 = new StringReader("{7(h!}Yz");
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 31, stringReader0, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(readerBasedJsonParser0, readerBasedJsonParser0);
      JsonToken jsonToken0 = jsonParserSequence0.nextToken();
      assertFalse(jsonToken0.isStructEnd());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      IOContext iOContext0 = new IOContext(bufferRecycler0, bufferRecycler0, true);
      CharsToNameCanonicalizer charsToNameCanonicalizer0 = CharsToNameCanonicalizer.createRoot();
      char[] charArray0 = new char[9];
      charArray0[2] = '4';
      ReaderBasedJsonParser readerBasedJsonParser0 = new ReaderBasedJsonParser(iOContext0, 2, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0, charArray0, 2, 3, false);
      ReaderBasedJsonParser readerBasedJsonParser1 = new ReaderBasedJsonParser(iOContext0, 3, (Reader) null, (ObjectCodec) null, charsToNameCanonicalizer0);
      JsonParserSequence jsonParserSequence0 = JsonParserSequence.createFlattened(readerBasedJsonParser1, readerBasedJsonParser0);
      jsonParserSequence0.nextToken();
      assertTrue(readerBasedJsonParser1.isClosed());
      assertEquals(4, readerBasedJsonParser0.getTokenColumnNr());
  }
}

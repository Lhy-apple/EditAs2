/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:10:11 GMT 2023
 */

package com.fasterxml.jackson.databind.node;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonLocation;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.core.Version;
import com.fasterxml.jackson.core.util.ByteArrayBuilder;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ext.CoreXMLSerializers;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.BinaryNode;
import com.fasterxml.jackson.databind.node.DecimalNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.LongNode;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.NumericNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.fasterxml.jackson.databind.node.TreeTraversingParser;
import com.fasterxml.jackson.databind.node.ValueNode;
import com.fasterxml.jackson.databind.util.RawValue;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TreeTraversingParser_ESTest extends TreeTraversingParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      NullNode nullNode0 = NullNode.getInstance();
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(nullNode0);
      treeTraversingParser0.setCodec((ObjectCodec) null);
      assertFalse(treeTraversingParser0.hasTextCharacters());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      NumericNode numericNode0 = jsonNodeFactory0.numberNode((short) (-1047));
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(numericNode0);
      Version version0 = treeTraversingParser0.version();
      assertFalse(version0.isUnknownVersion());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      float float0 = treeTraversingParser0.getFloatValue();
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(arrayNode0);
      try { 
        treeTraversingParser0.getNumberValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (null) not numeric, cannot use numeric value accessors
         //  at [Source: UNKNOWN; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      // Undeclared exception!
      try { 
        treeTraversingParser0.readValueAsTree();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No ObjectCodec defined for parser, needed for deserialization
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      boolean boolean0 = treeTraversingParser0.hasTextCharacters();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      // Undeclared exception!
      try { 
        treeTraversingParser0.getTextLength();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.node.TreeTraversingParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LongNode longNode0 = LongNode.valueOf(0L);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(longNode0, (ObjectCodec) null);
      int int0 = treeTraversingParser0.getTextOffset();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      boolean boolean0 = treeTraversingParser0.isClosed();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      BigInteger bigInteger0 = treeTraversingParser0.getBigIntegerValue();
      assertEquals((short)0, bigInteger0.shortValue());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(objectNode0);
      Object object0 = treeTraversingParser0.getCurrentValue();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      // Undeclared exception!
      try { 
        treeTraversingParser0._handleEOF();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Internal error: this code path should never get executed
         //
         verifyException("com.fasterxml.jackson.core.util.VersionUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      BinaryNode binaryNode0 = BinaryNode.valueOf(byteArray0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      JsonLocation jsonLocation0 = treeTraversingParser0.getTokenLocation();
      assertEquals((-1), jsonLocation0.getLineNr());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      short short0 = treeTraversingParser0.getShortValue();
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      // Undeclared exception!
      try { 
        treeTraversingParser0.getTextCharacters();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.node.TreeTraversingParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      BigDecimal bigDecimal0 = treeTraversingParser0.getDecimalValue();
      assertEquals((byte)0, bigDecimal0.byteValue());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BinaryNode binaryNode0 = BinaryNode.EMPTY_BINARY_NODE;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      try { 
        treeTraversingParser0.getLongValue();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Current token (VALUE_EMBEDDED_OBJECT) not numeric, cannot use numeric value accessors
         //  at [Source: UNKNOWN; line: -1, column: -1]
         //
         verifyException("com.fasterxml.jackson.core.JsonParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      double double0 = treeTraversingParser0.getDoubleValue();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(arrayNode0);
      treeTraversingParser0.close();
      treeTraversingParser0.close();
      assertTrue(treeTraversingParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      byte[] byteArray0 = new byte[1];
      BinaryNode binaryNode0 = arrayNode0.binaryNode(byteArray0, 0, 0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      treeTraversingParser0.nextToken();
      treeTraversingParser0.getText();
      assertEquals(12, treeTraversingParser0.currentTokenId());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(arrayNode0);
      treeTraversingParser0.nextToken();
      treeTraversingParser0.skipChildren();
      assertEquals(JsonToken.END_ARRAY, treeTraversingParser0.getCurrentToken());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      treeTraversingParser0._startContainer = true;
      treeTraversingParser0.nextToken();
      assertEquals(4, treeTraversingParser0.currentTokenId());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BinaryNode binaryNode0 = BinaryNode.EMPTY_BINARY_NODE;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      treeTraversingParser0.close();
      treeTraversingParser0.nextToken();
      assertTrue(treeTraversingParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TextNode textNode0 = new TextNode("Current token (");
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(textNode0);
      treeTraversingParser0.nextTextValue();
      treeTraversingParser0.nextToken();
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode(4);
      ObjectNode objectNode0 = arrayNode0.insertObject(5);
      objectNode0.set("", objectNode0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(objectNode0);
      treeTraversingParser0.nextToken();
      JsonToken jsonToken0 = treeTraversingParser0.nextToken();
      JsonToken jsonToken1 = treeTraversingParser0.nextToken();
      assertFalse(jsonToken1.equals((Object)jsonToken0));
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      ObjectNode objectNode0 = arrayNode0.objectNode();
      objectNode0.putArray("pW4");
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(objectNode0);
      treeTraversingParser0.nextValue();
      treeTraversingParser0.nextToken();
      JsonToken jsonToken0 = treeTraversingParser0.nextToken();
      assertFalse(jsonToken0.isScalarValue());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      byte[] byteArray0 = new byte[0];
      BinaryNode binaryNode0 = arrayNode0.binaryNode(byteArray0, 0, 0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      JsonParser jsonParser0 = treeTraversingParser0.skipChildren();
      assertFalse(jsonParser0.hasCurrentToken());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ObjectNode objectNode0 = new ObjectNode(jsonNodeFactory0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(objectNode0);
      treeTraversingParser0.nextToken();
      treeTraversingParser0.skipChildren();
      assertFalse(treeTraversingParser0.isExpectedStartObjectToken());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      String string0 = treeTraversingParser0.getCurrentName();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BinaryNode binaryNode0 = BinaryNode.EMPTY_BINARY_NODE;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      treeTraversingParser0.close();
      treeTraversingParser0.getCurrentName();
      assertTrue(treeTraversingParser0.isClosed());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      byte[] byteArray0 = new byte[0];
      BinaryNode binaryNode0 = arrayNode0.binaryNode(byteArray0, 0, 0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      treeTraversingParser0.nextToken();
      treeTraversingParser0.nextToken();
      treeTraversingParser0.overrideCurrentName("");
      assertEquals(0, treeTraversingParser0.currentTokenId());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode(4);
      ObjectNode objectNode0 = arrayNode0.insertObject(5);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(objectNode0);
      treeTraversingParser0.overrideCurrentName((String) null);
      assertEquals(0, treeTraversingParser0.getFeatureMask());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BinaryNode binaryNode0 = BinaryNode.EMPTY_BINARY_NODE;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      treeTraversingParser0.close();
      String string0 = treeTraversingParser0.getText();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      CoreXMLSerializers.XMLGregorianCalendarSerializer coreXMLSerializers_XMLGregorianCalendarSerializer0 = new CoreXMLSerializers.XMLGregorianCalendarSerializer();
      ObjectMapper objectMapper0 = new ObjectMapper((JsonFactory) null);
      SerializerProvider serializerProvider0 = objectMapper0.getSerializerProviderInstance();
      JsonNode jsonNode0 = coreXMLSerializers_XMLGregorianCalendarSerializer0.getSchema(serializerProvider0, (Type) null, false);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(jsonNode0);
      treeTraversingParser0.nextLongValue(1L);
      treeTraversingParser0.nextToken();
      String string0 = treeTraversingParser0.getText();
      assertEquals("type", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      LongNode longNode0 = LongNode.valueOf(0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(longNode0);
      treeTraversingParser0.nextToken();
      String string0 = treeTraversingParser0.getText();
      assertEquals("0", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      treeTraversingParser0.nextToken();
      String string0 = treeTraversingParser0.getText();
      assertEquals("0", string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.withExactBigDecimals(false);
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode(4);
      ObjectNode objectNode0 = arrayNode0.insertObject(5);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(objectNode0);
      treeTraversingParser0.nextToken();
      String string0 = treeTraversingParser0.getText();
      assertEquals("{", string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      byte[] byteArray0 = new byte[1];
      BinaryNode binaryNode0 = arrayNode0.binaryNode(byteArray0, 0, 0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      JsonToken jsonToken0 = treeTraversingParser0.nextToken();
      TreeTraversingParser treeTraversingParser1 = new TreeTraversingParser(arrayNode0);
      treeTraversingParser1._nextToken = jsonToken0;
      treeTraversingParser1.nextToken();
      String string0 = treeTraversingParser1.getText();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ValueNode valueNode0 = jsonNodeFactory0.rawValueNode((RawValue) null);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(valueNode0);
      treeTraversingParser0.nextToken();
      String string0 = treeTraversingParser0.getText();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      LongNode longNode0 = LongNode.valueOf(0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(longNode0);
      JsonParser.NumberType jsonParser_NumberType0 = treeTraversingParser0.getNumberType();
      assertEquals(JsonParser.NumberType.LONG, jsonParser_NumberType0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      BinaryNode binaryNode0 = BinaryNode.EMPTY_BINARY_NODE;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      treeTraversingParser0.close();
      Object object0 = treeTraversingParser0.getEmbeddedObject();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ArrayNode arrayNode0 = jsonNodeFactory0.arrayNode();
      byte[] byteArray0 = new byte[0];
      BinaryNode binaryNode0 = arrayNode0.binaryNode(byteArray0, 0, 0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      Object object0 = treeTraversingParser0.getEmbeddedObject();
      assertNotSame(byteArray0, object0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      treeTraversingParser0.nextToken();
      treeTraversingParser0.nextToken();
      Object object0 = treeTraversingParser0.getEmbeddedObject();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = JsonNodeFactory.instance;
      ValueNode valueNode0 = jsonNodeFactory0.rawValueNode((RawValue) null);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(valueNode0);
      Object object0 = treeTraversingParser0.getEmbeddedObject();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      JsonNodeFactory jsonNodeFactory0 = new JsonNodeFactory();
      ArrayNode arrayNode0 = new ArrayNode(jsonNodeFactory0);
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(arrayNode0);
      treeTraversingParser0.close();
      boolean boolean0 = treeTraversingParser0.isNaN();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      boolean boolean0 = treeTraversingParser0.isNaN();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      BinaryNode binaryNode0 = BinaryNode.EMPTY_BINARY_NODE;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      boolean boolean0 = treeTraversingParser0.isNaN();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      treeTraversingParser0.close();
      byte[] byteArray0 = treeTraversingParser0.getBinaryValue();
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TextNode textNode0 = new TextNode("");
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(textNode0);
      byte[] byteArray0 = treeTraversingParser0.getBinaryValue();
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      DecimalNode decimalNode0 = DecimalNode.ZERO;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(decimalNode0);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      int int0 = treeTraversingParser0.readBinaryValue((OutputStream) byteArrayBuilder0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      BinaryNode binaryNode0 = BinaryNode.EMPTY_BINARY_NODE;
      TreeTraversingParser treeTraversingParser0 = new TreeTraversingParser(binaryNode0);
      ByteArrayBuilder byteArrayBuilder0 = new ByteArrayBuilder();
      int int0 = treeTraversingParser0.readBinaryValue((OutputStream) byteArrayBuilder0);
      assertEquals(0, int0);
  }
}
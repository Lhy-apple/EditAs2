/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:31:36 GMT 2023
 */

package com.google.gson.internal.bind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.TypeAdapter;
import com.google.gson.TypeAdapterFactory;
import com.google.gson.internal.bind.TypeAdapters;
import com.google.gson.reflect.TypeToken;
import java.io.Reader;
import java.io.StringReader;
import java.lang.reflect.Type;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.URI;
import java.net.URL;
import java.util.BitSet;
import java.util.Calendar;
import java.util.Currency;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockInetAddress;
import org.evosuite.runtime.mock.java.net.MockURI;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.evosuite.runtime.mock.java.util.MockGregorianCalendar;
import org.evosuite.runtime.mock.java.util.MockUUID;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeAdapters_ESTest extends TypeAdapters_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Gson gson0 = new Gson();
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) mockGregorianCalendar0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Gson gson0 = new Gson();
      String string0 = gson0.toString();
      Class<JsonObject> class0 = JsonObject.class;
      TypeToken<JsonObject> typeToken0 = TypeToken.get(class0);
      Class<? super JsonObject> class1 = typeToken0.getRawType();
      try { 
        gson0.fromJson(string0, (Type) class1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Unterminated object at line 1 column 32 path $.serializeNulls
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Character> class0 = Character.class;
      TypeAdapter<Character> typeAdapter0 = TypeAdapters.CHARACTER;
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newFactoryForMultipleTypes(class0, (Class<? extends Character>) class0, (TypeAdapter<? super Character>) typeAdapter0);
      Class<Byte> class1 = Byte.class;
      TypeToken<Byte> typeToken0 = TypeToken.get(class1);
      Gson gson0 = new Gson();
      TypeAdapter<Byte> typeAdapter1 = gson0.getDelegateAdapter(typeAdapterFactory0, typeToken0);
      TypeAdapterFactory typeAdapterFactory1 = TypeAdapters.newFactory(typeToken0, typeAdapter1);
      assertFalse(typeAdapterFactory1.equals((Object)typeAdapterFactory0));
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TypeAdapter<String> typeAdapter0 = TypeAdapters.STRING;
      Class<String> class0 = String.class;
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newTypeHierarchyFactory(class0, typeAdapter0);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Gson gson0 = new Gson();
      Short short0 = new Short((short)237);
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) short0);
      assertTrue(jsonPrimitive0.isNumber());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Gson gson0 = new Gson();
      UUID uUID0 = MockUUID.randomUUID();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) uUID0);
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) jsonElement0);
      assertNotSame(jsonPrimitive0, jsonElement0);
      assertTrue(jsonPrimitive0.isString());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Gson gson0 = new Gson();
      Byte byte0 = new Byte((byte) (-21));
      JsonElement jsonElement0 = gson0.toJsonTree((Object) byte0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Gson gson0 = new Gson();
      AtomicBoolean atomicBoolean0 = new AtomicBoolean();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) atomicBoolean0);
      assertFalse(jsonElement0.isJsonObject());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Gson gson0 = new Gson();
      AtomicInteger atomicInteger0 = new AtomicInteger(5);
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) atomicInteger0);
      assertFalse(jsonPrimitive0.isBoolean());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Gson gson0 = new Gson();
      Locale locale0 = Locale.KOREA;
      Currency currency0 = Currency.getInstance(locale0);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) currency0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Currency> class0 = Currency.class;
      // Undeclared exception!
      try { 
        gson0.fromJson("$R6KcQ-_B7TVn'", class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Currency", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Character> class0 = Character.TYPE;
      // Undeclared exception!
      try { 
        gson0.toJsonTree((Object) class0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Attempted to serialize java.lang.Class: char. Forgot to register a type adapter?
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$5", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Gson gson0 = new Gson();
      Boolean boolean0 = new Boolean("0[jH\"\"5`");
      JsonPrimitive jsonPrimitive0 = new JsonPrimitive(boolean0);
      Class<BitSet> class0 = BitSet.class;
      try { 
        gson0.fromJson((JsonElement) jsonPrimitive0, class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected BEGIN_ARRAY but was BOOLEAN at path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Gson gson0 = new Gson();
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)32;
      BitSet bitSet0 = BitSet.valueOf(byteArray0);
      String string0 = gson0.toJson((Object) bitSet0);
      assertEquals("[0,0,0,0,0,1]", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Boolean> class0 = Boolean.TYPE;
      try { 
        gson0.fromJson("e')p4l&f'KpBLlzGT{", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 19 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Boolean> class0 = Boolean.TYPE;
      InetAddress inetAddress0 = gson0.fromJson("null", (Type) class0);
      assertNull(inetAddress0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonArray jsonArray0 = new JsonArray();
      Class<Boolean> class0 = Boolean.TYPE;
      try { 
        gson0.fromJson((JsonElement) jsonArray0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected BOOLEAN but was BEGIN_ARRAY at path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Byte> class0 = Byte.TYPE;
      try { 
        gson0.fromJson("d[*", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"d\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$9", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Short> class0 = Short.TYPE;
      try { 
        gson0.fromJson("g442y_#,Er,.]", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"g442y_\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$10", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Gson gson0 = new Gson();
      StringReader stringReader0 = new StringReader("null");
      Class<Short> class0 = Short.class;
      Short short0 = gson0.fromJson((Reader) stringReader0, class0);
      assertNull(short0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Integer> class0 = Integer.TYPE;
      try { 
        gson0.fromJson("0[jH\"\"5`", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 3 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Gson gson0 = new Gson();
      AtomicIntegerArray atomicIntegerArray0 = new AtomicIntegerArray(266);
      JsonArray jsonArray0 = (JsonArray)gson0.toJsonTree((Object) atomicIntegerArray0);
      assertEquals(266, jsonArray0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Long> class0 = Long.TYPE;
      try { 
        gson0.fromJson("closed", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"closed\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$12", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Long> class0 = Long.class;
      Long long0 = gson0.fromJson("null", class0);
      assertNull(long0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Character> class0 = Character.TYPE;
      try { 
        gson0.fromJson("+VY`tbasmmJj\"Ga1(", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Expecting character, got: +VY`tbasmmJj\"Ga1(
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$16", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Character> class0 = Character.TYPE;
      InetAddress inetAddress0 = gson0.fromJson("null", (Type) class0);
      assertNull(inetAddress0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Character> class0 = Character.TYPE;
      try { 
        gson0.fromJson("d[*", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 3 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Gson gson0 = new Gson();
      Character character0 = new Character('T');
      JsonElement jsonElement0 = gson0.toJsonTree((Object) character0);
      assertTrue(jsonElement0.isJsonPrimitive());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Character> class0 = Character.TYPE;
      JsonElement jsonElement0 = gson0.toJsonTree((Object) null, (Type) class0);
      assertFalse(jsonElement0.isJsonObject());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<String> class0 = String.class;
      try { 
        gson0.fromJson("Attempted to serialize java.lang.Class: ", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 12 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<String> class0 = String.class;
      InetAddress inetAddress0 = gson0.fromJson("null", (Type) class0);
      assertNull(inetAddress0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Gson gson0 = new Gson();
      StringBuilder stringBuilder0 = new StringBuilder();
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) stringBuilder0);
      assertFalse(jsonPrimitive0.isBoolean());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Gson gson0 = new Gson();
      StringBuilder stringBuilder0 = new StringBuilder();
      StringBuffer stringBuffer0 = new StringBuffer(stringBuilder0);
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) stringBuffer0);
      assertFalse(jsonPrimitive0.isNumber());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<URL> class0 = URL.class;
      try { 
        gson0.fromJson("Mw{N&8pm", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.net.MalformedURLException: no protocol: Mw
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonNull jsonNull0 = JsonNull.INSTANCE;
      Class<URL> class0 = URL.class;
      URL uRL0 = gson0.fromJson((JsonElement) jsonNull0, class0);
      assertNull(uRL0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Gson gson0 = new Gson();
      URL uRL0 = MockURL.getFileExample();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) uRL0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Gson gson0 = new Gson();
      URI uRI0 = MockURI.aHttpURI;
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) uRI0);
      assertFalse(jsonPrimitive0.isNumber());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<InetAddress> class0 = InetAddress.class;
      Inet4Address inet4Address0 = (Inet4Address)gson0.fromJson("h\"s", (Type) class0);
      assertFalse(inet4Address0.isMulticastAddress());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Gson gson0 = new Gson();
      InetAddress inetAddress0 = MockInetAddress.anyLocalAddress();
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) inetAddress0);
      assertFalse(jsonPrimitive0.isNumber());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Locale> class0 = Locale.class;
      try { 
        gson0.fromJson("Attempted to serialize java.lang.Class: ", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 12 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Gson gson0 = new Gson();
      Locale locale0 = Locale.JAPAN;
      JsonElement jsonElement0 = gson0.toJsonTree((Object) locale0);
      assertFalse(jsonElement0.isJsonObject());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<JsonObject> class0 = JsonObject.class;
      try { 
        gson0.fromJson("4=mu:;,WK^aUC{F[", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Expected a com.google.gson.JsonObject but was com.google.gson.JsonPrimitive
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$35$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<JsonObject> class0 = JsonObject.class;
      try { 
        gson0.fromJson("[hq`)^^jIa#<m]b'm", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.io.EOFException: End of input at line 1 column 18 path $[1]
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<JsonObject> class0 = JsonObject.class;
      TypeToken<JsonObject> typeToken0 = TypeToken.get(class0);
      Class<? super JsonObject> class1 = typeToken0.getRawType();
      try { 
        gson0.fromJson("null", (Type) class1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Expected a com.google.gson.JsonObject but was com.google.gson.JsonNull
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$35$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Gson gson0 = new Gson();
      String string0 = gson0.toJson((JsonElement) null);
      assertEquals("null", string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Gson gson0 = new Gson();
      String string0 = gson0.toJson((Object) null);
      assertEquals("null", string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Gson gson0 = new Gson();
      Double double0 = new Double(0.0);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) double0);
      // Undeclared exception!
      try { 
        gson0.toJson(jsonElement0, (Appendable) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.gson.internal.Streams$AppendableWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Gson gson0 = new Gson();
      Boolean boolean0 = new Boolean("com.google.gson.internal.bind.TypeAdapters$34");
      JsonPrimitive jsonPrimitive0 = new JsonPrimitive(boolean0);
      JsonPrimitive jsonPrimitive1 = (JsonPrimitive)gson0.toJsonTree((Object) jsonPrimitive0);
      assertTrue(jsonPrimitive1.isBoolean());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonObject jsonObject0 = new JsonObject();
      JsonArray jsonArray0 = new JsonArray();
      jsonObject0.add("Bwn*ODcI$cL1Ao<44", jsonArray0);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) jsonObject0);
      assertTrue(jsonElement0.equals((Object)jsonObject0));
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      JsonArray jsonArray0 = new JsonArray();
      jsonArray0.add((JsonElement) jsonArray0);
      // Undeclared exception!
      try { 
        jsonArray0.getAsJsonObject();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) gson0);
      assertTrue(jsonElement0.isJsonObject());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Calendar> class0 = Calendar.class;
      TypeAdapter<Calendar> typeAdapter0 = gson0.getAdapter(class0);
      assertNotNull(typeAdapter0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<GregorianCalendar> class0 = GregorianCalendar.class;
      TypeAdapter<GregorianCalendar> typeAdapter0 = gson0.getAdapter(class0);
      assertNotNull(typeAdapter0);
  }
}
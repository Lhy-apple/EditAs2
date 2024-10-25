/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:16:12 GMT 2023
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
import com.google.gson.internal.Excluder;
import com.google.gson.internal.bind.TypeAdapters;
import com.google.gson.reflect.TypeToken;
import java.io.CharArrayWriter;
import java.lang.reflect.Type;
import java.net.InetAddress;
import java.net.URI;
import java.net.URL;
import java.util.BitSet;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Locale;
import java.util.UUID;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.net.MockURI;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.evosuite.runtime.mock.java.util.MockGregorianCalendar;
import org.evosuite.runtime.mock.java.util.MockUUID;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeAdapters_ESTest extends TypeAdapters_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<URL> class0 = URL.class;
      TypeToken<URL> typeToken0 = TypeToken.get(class0);
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newFactory(typeToken0, (TypeAdapter<URL>) null);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Gson gson0 = new Gson();
      String string0 = gson0.toString();
      Class<GregorianCalendar> class0 = GregorianCalendar.class;
      try { 
        gson0.fromJson(string0, class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected an int but was STRING at line 1 column 17 path $.serializeNulls
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Gson gson0 = new Gson();
      Locale locale0 = Locale.JAPAN;
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar(locale0);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) mockGregorianCalendar0);
      assertFalse(jsonElement0.isJsonNull());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Gson gson0 = new Gson();
      Excluder excluder0 = new Excluder();
      Class<Object> class0 = Object.class;
      TypeToken<Object> typeToken0 = TypeToken.get(class0);
      TypeAdapter<Object> typeAdapter0 = gson0.getDelegateAdapter((TypeAdapterFactory) excluder0, typeToken0);
      Class<URL> class1 = URL.class;
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newFactoryForMultipleTypes(class1, (Class<? extends URL>) class1, (TypeAdapter<? super URL>) typeAdapter0);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newTypeHierarchyFactory(class0, (TypeAdapter<InetAddress>) null);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Short> class0 = Short.TYPE;
      Gson gson0 = new Gson();
      String string0 = gson0.toJson((Object) null, (Type) class0);
      assertEquals("null", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Gson gson0 = new Gson();
      Byte byte0 = new Byte((byte) (-118));
      JsonElement jsonElement0 = gson0.toJsonTree((Object) byte0);
      assertFalse(jsonElement0.isJsonNull());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Short> class0 = Short.TYPE;
      Gson gson0 = new Gson();
      // Undeclared exception!
      try { 
        gson0.toJson((Object) class0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Attempted to serialize java.lang.Class: short. Forgot to register a type adapter?
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonObject jsonObject0 = new JsonObject();
      Class<BitSet> class0 = BitSet.class;
      try { 
        gson0.fromJson((JsonElement) jsonObject0, class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected BEGIN_ARRAY but was BEGIN_OBJECT
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonNull jsonNull0 = JsonNull.INSTANCE;
      Class<BitSet> class0 = BitSet.class;
      BitSet bitSet0 = gson0.fromJson((JsonElement) jsonNull0, class0);
      assertNull(bitSet0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder();
      Gson gson0 = new Gson();
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte) (-10);
      BitSet bitSet0 = BitSet.valueOf(byteArray0);
      gson0.toJson((Object) bitSet0, (Appendable) stringBuilder0);
      assertEquals("[0,1,1,0,1,1,1,1]", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Boolean> class0 = Boolean.TYPE;
      Boolean boolean0 = gson0.fromJson("p", class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Boolean> class0 = Boolean.class;
      try { 
        gson0.fromJson("6", class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected a boolean but was NUMBER at line 1 column 2 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Boolean> class0 = Boolean.TYPE;
      JsonElement jsonElement0 = gson0.toJsonTree((Object) null, (Type) class0);
      Class<JsonObject> class1 = JsonObject.class;
      // Undeclared exception!
      try { 
        gson0.fromJson(jsonElement0, class1);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Byte> class0 = Byte.TYPE;
      try { 
        gson0.fromJson("_~`_RT~!G5i", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected an int but was STRING at line 1 column 1 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Byte> class0 = Byte.class;
      GregorianCalendar gregorianCalendar0 = gson0.fromJson("null", (Type) class0);
      assertNull(gregorianCalendar0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson("Error", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected an int but was STRING at line 1 column 1 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Short> class0 = Short.class;
      Gson gson0 = new Gson();
      GregorianCalendar gregorianCalendar0 = gson0.fromJson("null", (Type) class0);
      assertNull(gregorianCalendar0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson("l)<-[a<$yxkG1ox{76", class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected an int but was STRING at line 1 column 1 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Integer> class0 = Integer.class;
      Integer integer0 = gson0.fromJson("null", class0);
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Long> class0 = Long.TYPE;
      try { 
        gson0.fromJson("e!/A)5<ALymcT)fuKf", class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected a long but was STRING at line 1 column 1 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Long> class0 = Long.class;
      Long long0 = gson0.fromJson("null", class0);
      assertNull(long0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder();
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      Gson gson0 = new Gson();
      gson0.toJson((Object) charArrayWriter0, (Appendable) stringBuilder0);
      assertEquals(0, charArrayWriter0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Character> class0 = Character.TYPE;
      JsonElement jsonElement0 = gson0.toJsonTree((Object) null, (Type) class0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<String> class0 = String.class;
      try { 
        gson0.fromJson("Invalidbitset value type: ", class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 16 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder();
      Gson gson0 = new Gson();
      gson0.toJson((Object) stringBuilder0, (Appendable) stringBuilder0);
      assertEquals("\"\"", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<StringBuffer> class0 = StringBuffer.class;
      try { 
        gson0.fromJson("hx{#kkRQJwM", class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 4 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonNull jsonNull0 = JsonNull.INSTANCE;
      Class<StringBuffer> class0 = StringBuffer.class;
      StringBuffer stringBuffer0 = gson0.fromJson((JsonElement) jsonNull0, class0);
      assertNull(stringBuffer0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Gson gson0 = new Gson();
      StringBuffer stringBuffer0 = new StringBuffer();
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) stringBuffer0);
      assertFalse(jsonPrimitive0.isNumber());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<StringBuffer> class0 = StringBuffer.class;
      Gson gson0 = new Gson();
      String string0 = gson0.toJson((Object) null, (Type) class0);
      assertEquals("null", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<URL> class0 = URL.class;
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson("This is not a JSON Array.", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.net.MalformedURLException: no protocol: This
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Gson gson0 = new Gson();
      URL uRL0 = MockURL.getFileExample();
      // Undeclared exception!
      try { 
        gson0.toJson((Object) uRL0, (Appendable) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<URL> class0 = URL.class;
      Gson gson0 = new Gson();
      String string0 = gson0.toJson((Object) null, (Type) class0);
      assertEquals("null", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<URI> class0 = URI.class;
      try { 
        gson0.fromJson("ZjnfY;p}aV%eoT0y4)", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 7 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<URI> class0 = URI.class;
      Gson gson0 = new Gson();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) null, (Type) class0);
      URI uRI0 = gson0.fromJson(jsonElement0, class0);
      assertNull(uRI0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Gson gson0 = new Gson();
      URI uRI0 = MockURI.aFileURI;
      // Undeclared exception!
      try { 
        gson0.toJson((Object) uRI0, (Appendable) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JsonArray jsonArray0 = new JsonArray();
      Gson gson0 = new Gson();
      Class<InetAddress> class0 = InetAddress.class;
      try { 
        gson0.fromJson((JsonElement) jsonArray0, class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected STRING but was BEGIN_ARRAY
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonNull jsonNull0 = JsonNull.INSTANCE;
      Class<InetAddress> class0 = InetAddress.class;
      InetAddress inetAddress0 = gson0.fromJson((JsonElement) jsonNull0, class0);
      assertNull(inetAddress0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<UUID> class0 = UUID.class;
      UUID uUID0 = gson0.fromJson("cZ6", class0);
      assertEquals(16793600L, uUID0.getMostSignificantBits());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder();
      Gson gson0 = new Gson();
      UUID uUID0 = MockUUID.randomUUID();
      gson0.toJson((Object) uUID0, (Appendable) stringBuilder0);
      assertEquals("\"00000000-0100-4000-8200-000003000000\"", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Locale> class0 = Locale.class;
      try { 
        gson0.fromJson("mrj}b$DTn#SSTy7(`", class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 5 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Gson gson0 = new Gson();
      Locale locale0 = Locale.US;
      // Undeclared exception!
      try { 
        gson0.toJson((Object) locale0, (Appendable) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Gson gson0 = new Gson();
      Short short0 = Short.valueOf((short)2578);
      JsonPrimitive jsonPrimitive0 = new JsonPrimitive(short0);
      Class<JsonObject> class0 = JsonObject.class;
      // Undeclared exception!
      try { 
        gson0.fromJson((JsonElement) jsonPrimitive0, class0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Gson gson0 = new Gson();
      Boolean boolean0 = Boolean.FALSE;
      JsonPrimitive jsonPrimitive0 = new JsonPrimitive(boolean0);
      Class<JsonNull> class0 = JsonNull.class;
      // Undeclared exception!
      try { 
        gson0.fromJson((JsonElement) jsonPrimitive0, class0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<JsonNull> class0 = JsonNull.class;
      try { 
        gson0.fromJson("Y6m{LTJ-", class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 5 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonArray jsonArray0 = new JsonArray();
      Class<JsonObject> class0 = JsonObject.class;
      // Undeclared exception!
      try { 
        gson0.fromJson((JsonElement) jsonArray0, class0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Gson gson0 = new Gson();
      String string0 = gson0.toJson((JsonElement) null);
      assertEquals("null", string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) gson0);
      String string0 = gson0.toJson(jsonElement0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Calendar> class0 = Calendar.class;
      TypeAdapter<Calendar> typeAdapter0 = gson0.getAdapter(class0);
      assertNotNull(typeAdapter0);
  }
}

/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:34:54 GMT 2023
 */

package com.google.gson.internal.bind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.TypeAdapter;
import com.google.gson.TypeAdapterFactory;
import com.google.gson.internal.bind.TypeAdapters;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.lang.reflect.Type;
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
import org.evosuite.runtime.ViolatedAssumptionAnswer;
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
      MockGregorianCalendar mockGregorianCalendar0 = new MockGregorianCalendar((-1929), 125, (-1929));
      gson0.toJsonTree((Object) mockGregorianCalendar0);
      assertEquals("org.evosuite.runtime.mock.java.util.MockGregorianCalendar[time=?,areFieldsSet=false,areAllFieldsSet=false,lenient=true,zone=sun.util.calendar.ZoneInfo[id=\"GMT\",offset=0,dstSavings=0,useDaylight=false,transitions=0,lastRule=null],firstDayOfWeek=1,minimalDaysInFirstWeek=1,ERA=?,YEAR=-1929,MONTH=125,WEEK_OF_YEAR=?,WEEK_OF_MONTH=?,DAY_OF_MONTH=-1929,DAY_OF_YEAR=?,DAY_OF_WEEK=?,DAY_OF_WEEK_IN_MONTH=?,AM_PM=0,HOUR=0,HOUR_OF_DAY=0,MINUTE=0,SECOND=0,MILLISECOND=?,ZONE_OFFSET=?,DST_OFFSET=?]", mockGregorianCalendar0.toString());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Gson gson0 = new Gson();
      String string0 = gson0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeAdapter<Character> typeAdapter0 = TypeAdapters.CHARACTER;
      Class<Character> class0 = Character.class;
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newFactoryForMultipleTypes(class0, (Class<? extends Character>) class0, (TypeAdapter<? super Character>) typeAdapter0);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Locale> class0 = Locale.class;
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newTypeHierarchyFactory(class0, (TypeAdapter<Locale>) null);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<URL> class0 = URL.class;
      TypeToken<URL> typeToken0 = TypeToken.get(class0);
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newFactory(typeToken0, (TypeAdapter<URL>) null);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Gson gson0 = new Gson();
      Short short0 = new Short((short)1200);
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) short0);
      assertTrue(jsonPrimitive0.isNumber());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) gson0);
      JsonElement jsonElement1 = gson0.toJsonTree((Object) jsonElement0);
      assertTrue(jsonElement1.equals((Object)jsonElement0));
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Gson gson0 = new Gson();
      Byte byte0 = new Byte((byte)35);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) byte0);
      assertTrue(jsonElement0.isJsonPrimitive());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Gson gson0 = new Gson();
      StringReader stringReader0 = new StringReader("allocateInstance");
      JsonReader jsonReader0 = new JsonReader(stringReader0);
      Class<AtomicBoolean> class0 = AtomicBoolean.class;
      TypeToken<AtomicBoolean> typeToken0 = TypeToken.get(class0);
      Class<? super AtomicBoolean> class1 = typeToken0.getRawType();
      try { 
        gson0.fromJson(jsonReader0, (Type) class1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected a boolean but was STRING at line 1 column 1 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Gson gson0 = new Gson();
      AtomicBoolean atomicBoolean0 = new AtomicBoolean();
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) atomicBoolean0);
      assertFalse(jsonPrimitive0.isString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Gson gson0 = new Gson();
      AtomicInteger atomicInteger0 = new AtomicInteger(49);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) atomicInteger0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Gson gson0 = new Gson();
      Locale locale0 = Locale.ITALY;
      Currency currency0 = Currency.getInstance(locale0);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) currency0);
      assertFalse(jsonElement0.isJsonNull());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Currency> class0 = Currency.class;
      Gson gson0 = new Gson();
      JsonArray jsonArray0 = new JsonArray();
      try { 
        gson0.fromJson((JsonElement) jsonArray0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected STRING but was BEGIN_ARRAY at path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonDeserializer<Integer> jsonDeserializer0 = (JsonDeserializer<Integer>) mock(JsonDeserializer.class, new ViolatedAssumptionAnswer());
      Gson gson0 = new Gson();
      // Undeclared exception!
      try { 
        gson0.toJsonTree((Object) jsonDeserializer0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Attempted to serialize java.lang.Class: com.google.gson.JsonDeserializer. Forgot to register a type adapter?
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$5", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Gson gson0 = new Gson();
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte) (-7);
      BitSet bitSet0 = BitSet.valueOf(byteArray0);
      JsonArray jsonArray0 = (JsonArray)gson0.toJsonTree((Object) bitSet0);
      assertEquals(8, jsonArray0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Boolean> class0 = Boolean.TYPE;
      try { 
        gson0.fromJson("Unterminated object", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 15 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Gson gson0 = new Gson();
      StringReader stringReader0 = new StringReader("null");
      JsonReader jsonReader0 = new JsonReader(stringReader0);
      Class<Boolean> class0 = Boolean.TYPE;
      Byte byte0 = gson0.fromJson(jsonReader0, (Type) class0);
      assertNull(byte0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StringReader stringReader0 = new StringReader("null");
      JsonReader jsonReader0 = new JsonReader(stringReader0);
      Class<Boolean> class0 = Boolean.TYPE;
      Gson gson0 = new Gson();
      jsonReader0.skipValue();
      try { 
        gson0.fromJson(jsonReader0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected a boolean but was END_DOCUMENT at line 1 column 5 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Gson gson0 = new Gson();
      URL uRL0 = MockURL.getFileExample();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) uRL0);
      Class<Byte> class0 = Byte.TYPE;
      try { 
        gson0.fromJson(jsonElement0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"file://some/fake/but/wellformed/url\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$9", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonNull jsonNull0 = JsonNull.INSTANCE;
      Class<Byte> class0 = Byte.TYPE;
      Currency currency0 = gson0.fromJson((JsonElement) jsonNull0, (Type) class0);
      assertNull(currency0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Gson gson0 = new Gson();
      StringReader stringReader0 = new StringReader("allocateInstance");
      JsonReader jsonReader0 = new JsonReader(stringReader0);
      Class<Short> class0 = Short.TYPE;
      try { 
        gson0.fromJson(jsonReader0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"allocateInstance\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$10", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Integer> class0 = Integer.TYPE;
      StringReader stringReader0 = new StringReader("HBe8");
      try { 
        gson0.fromJson((Reader) stringReader0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"HBe8\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$11", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Gson gson0 = new Gson();
      AtomicIntegerArray atomicIntegerArray0 = new AtomicIntegerArray(190);
      JsonArray jsonArray0 = (JsonArray)gson0.toJsonTree((Object) atomicIntegerArray0);
      assertEquals(190, jsonArray0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Gson gson0 = new Gson();
      StringReader stringReader0 = new StringReader("+");
      JsonReader jsonReader0 = new JsonReader(stringReader0);
      Class<Long> class0 = Long.TYPE;
      try { 
        gson0.fromJson(jsonReader0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"+\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$12", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Gson gson0 = new Gson();
      StringReader stringReader0 = new StringReader("V4!}%");
      JsonReader jsonReader0 = new JsonReader(stringReader0);
      Class<Character> class0 = Character.TYPE;
      try { 
        gson0.fromJson(jsonReader0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Expecting character, got: V4!
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$16", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<Character> class0 = Character.TYPE;
      Gson gson0 = new Gson();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) null);
      JsonObject jsonObject0 = gson0.fromJson(jsonElement0, (Type) class0);
      assertNull(jsonObject0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Gson gson0 = new Gson();
      Character character0 = Character.valueOf('/');
      JsonElement jsonElement0 = gson0.toJsonTree((Object) character0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Gson gson0 = new Gson();
      StringBuilder stringBuilder0 = new StringBuilder(3921);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) stringBuilder0);
      assertTrue(jsonElement0.isJsonPrimitive());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Gson gson0 = new Gson();
      Long long0 = new Long(14);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) long0);
      Class<StringBuffer> class0 = StringBuffer.class;
      StringBuffer stringBuffer0 = gson0.fromJson(jsonElement0, class0);
      assertEquals(25, stringBuffer0.length());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<InetAddress> class0 = InetAddress.class;
      TypeAdapter<InetAddress> typeAdapter0 = gson0.getAdapter(class0);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) typeAdapter0);
      Class<StringBuffer> class1 = StringBuffer.class;
      StringBuffer stringBuffer0 = gson0.fromJson(jsonElement0, class1);
      assertNull(stringBuffer0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      Gson gson0 = new Gson();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) stringWriter0);
      assertTrue(jsonElement0.isJsonObject());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Gson gson0 = new Gson();
      StringReader stringReader0 = new StringReader("HBe8");
      Class<URL> class0 = URL.class;
      TypeToken<URL> typeToken0 = TypeToken.get(class0);
      Type type0 = typeToken0.getType();
      try { 
        gson0.fromJson((Reader) stringReader0, type0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.net.MalformedURLException: no protocol: HBe8
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonNull jsonNull0 = JsonNull.INSTANCE;
      Class<URL> class0 = URL.class;
      TypeToken<URL> typeToken0 = TypeToken.get(class0);
      Type type0 = typeToken0.getType();
      Currency currency0 = gson0.fromJson((JsonElement) jsonNull0, type0);
      assertNull(currency0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Gson gson0 = new Gson();
      URI uRI0 = MockURI.URI("com.google.gson.internal.bind.TypeAdapters$34");
      JsonElement jsonElement0 = gson0.toJsonTree((Object) uRI0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson("name == null", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 7 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Gson gson0 = new Gson();
      InetAddress inetAddress0 = MockInetAddress.anyLocalAddress();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) inetAddress0);
      assertTrue(jsonElement0.isJsonPrimitive());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Gson gson0 = new Gson();
      UUID uUID0 = MockUUID.randomUUID();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) uUID0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Class<Calendar> class0 = Calendar.class;
      StringReader stringReader0 = new StringReader("aklocat\"Instance");
      JsonReader jsonReader0 = new JsonReader(stringReader0);
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson(jsonReader0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected BEGIN_OBJECT but was STRING at line 1 column 1 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Gson gson0 = new Gson();
      Locale locale0 = Locale.ENGLISH;
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) locale0);
      assertFalse(jsonPrimitive0.isNumber());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Class<JsonObject> class0 = JsonObject.class;
      StringReader stringReader0 = new StringReader("allocateInstance");
      JsonReader jsonReader0 = new JsonReader(stringReader0);
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson(jsonReader0, (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Expected a com.google.gson.JsonObject but was com.google.gson.JsonPrimitive
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$35$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Gson gson0 = new Gson();
      // Undeclared exception!
      try { 
        gson0.toJson((JsonElement) null, (Appendable) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.gson.internal.Streams$AppendableWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<GregorianCalendar> class0 = GregorianCalendar.class;
      TypeToken<GregorianCalendar> typeToken0 = TypeToken.get(class0);
      TypeAdapter<GregorianCalendar> typeAdapter0 = gson0.getAdapter(typeToken0);
      assertNotNull(typeAdapter0);
  }
}
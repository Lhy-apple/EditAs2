/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:51:56 GMT 2023
 */

package com.google.gson.internal.bind;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializer;
import com.google.gson.TypeAdapter;
import com.google.gson.TypeAdapterFactory;
import com.google.gson.internal.bind.ObjectTypeAdapter;
import com.google.gson.internal.bind.TypeAdapters;
import com.google.gson.reflect.TypeToken;
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
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.evosuite.runtime.mock.java.net.MockInetAddress;
import org.evosuite.runtime.mock.java.net.MockURI;
import org.evosuite.runtime.mock.java.net.MockURL;
import org.evosuite.runtime.mock.java.util.MockUUID;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeAdapters_ESTest extends TypeAdapters_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Gson gson0 = new Gson();
      Long long0 = new Long(10L);
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) long0);
      assertFalse(jsonPrimitive0.isBoolean());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Gson gson0 = new Gson();
      String string0 = gson0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<String> class0 = String.class;
      Gson gson0 = new Gson();
      ObjectTypeAdapter objectTypeAdapter0 = new ObjectTypeAdapter(gson0);
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newFactoryForMultipleTypes(class0, (Class<? extends String>) class0, (TypeAdapter<? super String>) objectTypeAdapter0);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newTypeHierarchyFactory(class0, (TypeAdapter<InetAddress>) null);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<URI> class0 = URI.class;
      TypeToken<URI> typeToken0 = TypeToken.get(class0);
      TypeAdapter<URI> typeAdapter0 = TypeAdapters.URI;
      TypeAdapterFactory typeAdapterFactory0 = TypeAdapters.newFactory(typeToken0, typeAdapter0);
      assertNotNull(typeAdapterFactory0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Gson gson0 = new Gson();
      Short short0 = new Short((short)2758);
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) short0);
      assertTrue(jsonPrimitive0.isNumber());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Gson gson0 = new Gson();
      ObjectTypeAdapter objectTypeAdapter0 = new ObjectTypeAdapter(gson0);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) objectTypeAdapter0);
      Class<Calendar> class0 = Calendar.class;
      try { 
        gson0.fromJson(jsonElement0, class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected NUMBER but was BEGIN_OBJECT at path $.gson
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<InetAddress> class0 = InetAddress.class;
      Gson gson0 = new Gson();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) null, (Type) class0);
      String string0 = gson0.fromJson(jsonElement0, (Type) class0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) "*YG^xhk)'Wa");
      assertTrue(jsonPrimitive0.isString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Gson gson0 = new Gson();
      Byte byte0 = new Byte((byte)8);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) byte0);
      assertTrue(jsonElement0.isJsonPrimitive());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<AtomicBoolean> class0 = AtomicBoolean.class;
      TypeToken<AtomicBoolean> typeToken0 = TypeToken.get(class0);
      Type type0 = typeToken0.getType();
      try { 
        gson0.fromJson("2U{c<PB#(OfF", type0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected a boolean but was STRING at line 1 column 1 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Gson gson0 = new Gson();
      AtomicBoolean atomicBoolean0 = new AtomicBoolean();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) atomicBoolean0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Gson gson0 = new Gson();
      AtomicInteger atomicInteger0 = new AtomicInteger(2260);
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) atomicInteger0);
      assertFalse(jsonPrimitive0.isString());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Gson gson0 = new Gson();
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      Currency currency0 = Currency.getInstance(locale0);
      JsonElement jsonElement0 = gson0.toJsonTree((Object) currency0);
      assertFalse(jsonElement0.isJsonObject());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Currency> class0 = Currency.class;
      // Undeclared exception!
      try { 
        gson0.fromJson("Nf|!", class0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Currency", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonSerializer<Character> jsonSerializer0 = (JsonSerializer<Character>) mock(JsonSerializer.class, new ViolatedAssumptionAnswer());
      Gson gson0 = new Gson();
      MockPrintStream mockPrintStream0 = new MockPrintStream(")]}'\n");
      // Undeclared exception!
      try { 
        gson0.toJson((Object) jsonSerializer0, (Appendable) mockPrintStream0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // Attempted to serialize java.lang.Class: com.google.gson.JsonSerializer. Forgot to register a type adapter?
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$5", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Gson gson0 = new Gson();
      BitSet bitSet0 = new BitSet();
      bitSet0.set(581);
      JsonArray jsonArray0 = (JsonArray)gson0.toJsonTree((Object) bitSet0);
      assertEquals(582, jsonArray0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<Boolean> class0 = Boolean.TYPE;
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson("B)9(-}JI", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 7 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Byte> class0 = Byte.TYPE;
      try { 
        gson0.fromJson("mBtA`pRy._Ax#~6EBwC", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"mBtA`pRy._Ax\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$9", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Class<Short> class0 = Short.TYPE;
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson("com.google.gson.internal.bind.TypeAdapters$9", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"com.google.gson.internal.bind.TypeAdapters$9\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$10", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Integer> class0 = Integer.TYPE;
      try { 
        gson0.fromJson("mZcR@FMsI^,`H!r", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"mZcR@FMsI^\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$11", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Gson gson0 = new Gson();
      AtomicIntegerArray atomicIntegerArray0 = new AtomicIntegerArray(545);
      JsonArray jsonArray0 = (JsonArray)gson0.toJsonTree((Object) atomicIntegerArray0);
      assertEquals(545, jsonArray0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Long> class0 = Long.TYPE;
      try { 
        gson0.fromJson("HY<+a_a", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NumberFormatException: For input string: \"HY<+a_a\"
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$12", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Character> class0 = Character.TYPE;
      try { 
        gson0.fromJson("com.google.gson.internal.bind.TypeAdapters$22", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Expecting character, got: com.google.gson.internal.bind.TypeAdapters$22
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$16", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Character> class0 = Character.TYPE;
      try { 
        gson0.fromJson("g k/I1,hl31{", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 4 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Gson gson0 = new Gson();
      Character character0 = Character.valueOf('V');
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) character0);
      assertFalse(jsonPrimitive0.isNumber());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<Character> class0 = Character.TYPE;
      // Undeclared exception!
      try { 
        gson0.toJson((Object) null, (Type) class0, (Appendable) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.gson.internal.Streams$AppendableWriter", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<String> class0 = String.class;
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson("mZcR@FMsI^,`H!r", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 12 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<StringBuilder> class0 = StringBuilder.class;
      try { 
        gson0.fromJson("o<~;5!", (Type) class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 5 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<StringBuffer> class0 = StringBuffer.class;
      try { 
        gson0.fromJson("Z1AsR@}4}k/sSWyP2\"", class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Use JsonReader.setLenient(true) to accept malformed JSON at line 1 column 8 path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Gson gson0 = new Gson();
      StringBuffer stringBuffer0 = new StringBuffer("kMP\u0000C?Z7GJQTz");
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) stringBuffer0);
      assertFalse(jsonPrimitive0.isBoolean());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Gson gson0 = new Gson();
      URL uRL0 = MockURL.getFileExample();
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) uRL0);
      assertFalse(jsonPrimitive0.isBoolean());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Gson gson0 = new Gson();
      URI uRI0 = MockURI.aFTPURI;
      JsonPrimitive jsonPrimitive0 = (JsonPrimitive)gson0.toJsonTree((Object) uRI0);
      assertFalse(jsonPrimitive0.isBoolean());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Gson gson0 = new Gson();
      InetAddress inetAddress0 = MockInetAddress.getLocalHost();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) inetAddress0);
      assertFalse(jsonElement0.isJsonNull());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<UUID> class0 = UUID.class;
      UUID uUID0 = gson0.fromJson("com.google.gson.internal.JsonReaderInternalAccess", (Type) class0);
      assertEquals("00000000-0100-4000-8200-000003000000", uUID0.toString());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Gson gson0 = new Gson();
      Class<UUID> class0 = UUID.class;
      UUID uUID0 = gson0.fromJson("null", class0);
      assertNull(uUID0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Gson gson0 = new Gson();
      UUID uUID0 = MockUUID.randomUUID();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) uUID0);
      assertTrue(jsonElement0.isJsonPrimitive());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonElement jsonElement0 = gson0.toJsonTree((Object) null);
      Class<Calendar> class0 = Calendar.class;
      Calendar calendar0 = gson0.fromJson(jsonElement0, class0);
      assertNull(calendar0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonObject jsonObject0 = new JsonObject();
      Class<Calendar> class0 = Calendar.class;
      Calendar calendar0 = gson0.fromJson((JsonElement) jsonObject0, class0);
      assertEquals("org.evosuite.runtime.mock.java.util.MockGregorianCalendar[time=?,areFieldsSet=false,areAllFieldsSet=false,lenient=true,zone=sun.util.calendar.ZoneInfo[id=\"GMT\",offset=0,dstSavings=0,useDaylight=false,transitions=0,lastRule=null],firstDayOfWeek=1,minimalDaysInFirstWeek=1,ERA=?,YEAR=0,MONTH=0,WEEK_OF_YEAR=?,WEEK_OF_MONTH=?,DAY_OF_MONTH=0,DAY_OF_YEAR=?,DAY_OF_WEEK=?,DAY_OF_WEEK_IN_MONTH=?,AM_PM=0,HOUR=0,HOUR_OF_DAY=0,MINUTE=0,SECOND=0,MILLISECOND=?,ZONE_OFFSET=?,DST_OFFSET=?]", calendar0.toString());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      JsonArray jsonArray0 = new JsonArray();
      Gson gson0 = new Gson();
      Class<Locale> class0 = Locale.class;
      try { 
        gson0.fromJson((JsonElement) jsonArray0, class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Expected STRING but was BEGIN_ARRAY at path $
         //
         verifyException("com.google.gson.Gson", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Gson gson0 = new Gson();
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JsonElement jsonElement0 = gson0.toJsonTree((Object) locale0);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Class<JsonArray> class0 = JsonArray.class;
      TypeToken<JsonArray> typeToken0 = TypeToken.get(class0);
      Class<? super JsonArray> class1 = typeToken0.getRawType();
      Gson gson0 = new Gson();
      try { 
        gson0.fromJson("mZcR@FMsI^,`H!r", (Type) class1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Expected a com.google.gson.JsonArray but was com.google.gson.JsonPrimitive
         //
         verifyException("com.google.gson.internal.bind.TypeAdapters$35$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Gson gson0 = new Gson();
      String string0 = gson0.toJson((JsonElement) null);
      assertEquals("null", string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Gson gson0 = new Gson();
      String string0 = gson0.toJson((Object) null);
      assertEquals("null", string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Gson gson0 = new Gson();
      JsonObject jsonObject0 = new JsonObject();
      Character character0 = new Character('h');
      jsonObject0.addProperty("haM(hu)yuI#\"R~Jmz", character0);
      String string0 = gson0.toJson((JsonElement) jsonObject0);
      assertEquals("{\"haM(hu)yuI#\\\"R~Jmz\":\"h\"}", string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Gson gson0 = new Gson();
      Float float0 = new Float(0.0);
      JsonPrimitive jsonPrimitive0 = new JsonPrimitive(float0);
      String string0 = gson0.toJson((JsonElement) jsonPrimitive0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Gson gson0 = new Gson();
      Boolean boolean0 = Boolean.TRUE;
      JsonPrimitive jsonPrimitive0 = new JsonPrimitive(boolean0);
      JsonPrimitive jsonPrimitive1 = (JsonPrimitive)gson0.toJsonTree((Object) jsonPrimitive0);
      assertTrue(jsonPrimitive1.isBoolean());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      JsonPrimitive jsonPrimitive0 = new JsonPrimitive("YY)s34ge!*v.d");
      JsonArray jsonArray0 = new JsonArray();
      jsonArray0.add((JsonElement) jsonPrimitive0);
      // Undeclared exception!
      try { 
        jsonArray0.getAsJsonObject();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Not a JSON Object: [\"YY)s34ge!*v.d\"]
         //
         verifyException("com.google.gson.JsonElement", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Gson gson0 = new Gson();
      ObjectTypeAdapter objectTypeAdapter0 = new ObjectTypeAdapter(gson0);
      Class<GregorianCalendar> class0 = GregorianCalendar.class;
      TypeToken<GregorianCalendar> typeToken0 = TypeToken.get(class0);
      TypeAdapter<GregorianCalendar> typeAdapter0 = gson0.getDelegateAdapter(objectTypeAdapter0.FACTORY, typeToken0);
      assertNotNull(typeAdapter0);
  }
}
